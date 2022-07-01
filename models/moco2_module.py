# source: https://github.com/ElementAI/seasonal-contrast

from argparse import ArgumentParser
from itertools import chain

import torch
from torch import nn, optim
from torch import Tensor
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule
from pl_bolts.metrics import precision_at_k
from pl_bolts.models.self_supervised.moco.transforms import GaussianBlur
from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomHorizontalFlip, RandomThinPlateSpline, Normalize, RandomHorizontalFlip, RandomGrayscale, RandomGaussianBlur, RandomGaussianNoise


class DataAugmentationRGB(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_img_trf: bool = False, apply_color_jitter: bool = False, apply_gauss_noise: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter
        self._apply_gauss_noise = apply_gauss_noise
        self._apply_img_trf = apply_img_trf
        
#         self.normalize = Normalize([0.4194, 0.4505, 0.4099], [0.20, 0.1759, 0.1694])
        
        self.img_transforms = nn.Sequential(
            RandomHorizontalFlip(p=0.75),
#             RandomGaussianBlur((3, 3), (1., 2.0), p=0.5),
#             RandomChannelShuffle(p=0.75),
#             RandomGrayscale(p=0.2),
        )

        self.gaussian_noise = RandomGaussianNoise(mean=0., std=1., p=0.5)
        self.jitter = ColorJitter(0.4, 0.4, 0.4, 0.1)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        
        x_out = None
        if self._apply_img_trf:
            x_out = self.img_transforms(x)  # BxCxHxW
#             x_out = self.jitter(x_out)
        elif self._apply_color_jitter:
            x_out = self.jitter(x)
        elif self._apply_gauss_noise:
            x_out = self.gaussian_noise(x)
        return x_out
    
    

class MocoV2(LightningModule):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self.transforms_img = DataAugmentationRGB(apply_img_trf=True)
        self.transforms_jit = DataAugmentationRGB(apply_color_jitter=True)
        self.transforms_gauss = DataAugmentationRGB(apply_gauss_noise=True)
        
        self.emb_spaces=opts.ssl.num_keys
        self.base_encoder = opts.ssl.base_encoder
        self.emb_dim = opts.ssl.emb_dim
        self.num_negatives = opts.ssl.num_negatives
        self.encoder_momentum = opts.ssl.encoder_momentum
        self.softmax_temperature = opts.ssl.softmax_temperature
        self.learning_rate = opts.ssl.learning_rate            
        self.momentum = opts.ssl.momentum
        self.weight_decay = opts.ssl.weight_decay
        self.use_ddp = opts.ssl.use_ddp
        self.use_ddp2 = opts.ssl.use_ddp2

        # create the encoders
        template_model = getattr(torchvision.models, self.base_encoder)
        
        # load the same resnet50 random initialization
        self.encoder_q = template_model(pretrained=opts.ssl.ssl_pretrained)
        random_init_path = opts.random_init_path
        checkpoint = torch.load(random_init_path)
        self.encoder_q.load_state_dict(checkpoint['model_state_dict'])

        self.encoder_k = template_model(pretrained=opts.ssl.ssl_pretrained)
        self.encoder_q.fc = nn.Linear(512, self.emb_dim)
        self.encoder_k.fc = nn.Linear(512, self.emb_dim)
#             self.encoder_q = template_model(num_classes=self.emb_dim)
#             self.encoder_k = template_model(num_classes=self.emb_dim)

        # remove fc layer
        self.encoder_q = nn.Sequential(
            *list(self.encoder_q.children())[:-1], nn.Flatten()
        )
        self.encoder_k = nn.Sequential(
            *list(self.encoder_k.children())[:-1], nn.Flatten()
        )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the projection heads
        self.mlp_dim = 512 * (1 if self.base_encoder in ["resnet18", "resnet34"] else 4)
        self.heads_q = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.mlp_dim, self.mlp_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_dim, self.emb_dim),
                )
                for _ in range(self.emb_spaces)
            ]
        )
        self.heads_k = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.mlp_dim, self.mlp_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_dim, self.emb_dim),
                )
                for _ in range(self.emb_spaces)
            ]
        )

        for param_q, param_k in zip(
            self.heads_q.parameters(), self.heads_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.emb_spaces, self.emb_dim, self.num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(self.emb_spaces, 1, dtype=torch.long))

        
#     augment = transforms.Compose([
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
#             transforms.RandomHorizontalFlip(),
#         ])

    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        patches = batch
        
        # (k0, k1) -> temporal/img_trf
        # (k0, k2) -> synthtic/gauss

        rgb_img = patches['rgb']
        q = rgb_img
        k0 = self.transforms_jit(rgb_img)
        k1 = self.transforms_img(rgb_img)
        k2 = self.transforms_gauss(rgb_img)
        
#         q = self.preprocess_rgb(q)
#         k0 = self.preprocess_nearir(k0)
#         k1 = self.preprocess_rgb(k1)
#         k2 = self.preprocess_rgb(k2)
        
        return q, [k0, k1, k2]
    
    
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            em = self.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)
        for param_q, param_k in zip(
            self.heads_q.parameters(), self.heads_k.parameters()
        ):
            em = self.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_idx):
        # gather keys before updating queue
        if self.use_ddp or self.use_ddp2:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[queue_idx])
        assert self.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[queue_idx, :, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.num_negatives  # move pointer

        self.queue_ptr[queue_idx] = ptr

    def forward(self, img_q, img_k):
        """
        Input:
            img_q: a batch of query images
            img_k: a batch of key images
        Output:
            logits, targets
        """

        # update the key encoder
        self._momentum_update_key_encoder()

        # compute query features
        v_q = self.encoder_q(img_q)

        # compute key features
        v_k = []
        for i in range(self.emb_spaces):
            # shuffle for making use of BN
            if self.use_ddp or self.use_ddp2:
                img_k[i], idx_unshuffle = batch_shuffle_ddp(img_k[i])

            with torch.no_grad():  # no gradient to keys
                v_k.append(self.encoder_k(img_k[i]))

            # undo shuffle
            if self.use_ddp or self.use_ddp2:
                v_k[i] = batch_unshuffle_ddp(v_k[i], idx_unshuffle)

        logits = []
        for i in range(self.emb_spaces):
            # compute query projections
            z_q = self.heads_q[i](v_q)  # queries: NxC
            z_q = nn.functional.normalize(z_q, dim=1)

            # compute key projections
            z_k = []
            for j in range(self.emb_spaces):
                with torch.no_grad():  # no gradient to keys
                    z_k.append(self.heads_k[i](v_k[j]))  # keys: NxC
                    z_k[j] = nn.functional.normalize(z_k[j], dim=1)

            # select positive and negative pairs
            z_pos = z_k[i]
            z_neg = self.queue[i].clone().detach()
            if i > 0:  # embedding space 0 is invariant to all augmentations
                z_neg = torch.cat(
                    [
                        z_neg,
                        *[z_k[j].T for j in range(self.emb_spaces) if j != i],
                    ],
                    dim=1,
                )

            # compute logits
            # Einstein sum is more intuitive
            l_pos = torch.einsum("nc,nc->n", z_q, z_pos).unsqueeze(
                -1
            )  # positive logits: Nx1
            l_neg = torch.einsum("nc,ck->nk", z_q, z_neg)  # negative logits: NxK

            l = torch.cat([l_pos, l_neg], dim=1)  # logits: Nx(1+K)
            l /= self.softmax_temperature  # apply temperature
            logits.append(l)

            # dequeue and enqueue
            self._dequeue_and_enqueue(z_k[i], queue_idx=i)

        # targets: positive key indicators
        targets = torch.zeros(logits[0].shape[0], dtype=torch.long)
        targets = targets.type_as(logits[0])

        return logits, targets

    def training_step(self, batch, batch_idx):
        img_q, img_k = batch
        if self.emb_spaces == 1 and isinstance(img_k, torch.Tensor):
            img_k = [img_k]

        output, target = self(img_q, img_k)

        losses = []
        accuracies = []
        for out in output:
            losses.append(F.cross_entropy(out.float(), target.long()))
            accuracies.append(precision_at_k(out, target, top_k=(1,))[0])
        loss = torch.sum(torch.stack(losses))

        log = {"train_loss": loss}
        for i, acc in enumerate(accuracies):
            log[f"train_acc/subspace{i}"] = acc

        self.log_dict(log, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = chain(self.encoder_q.parameters(), self.heads_q.parameters())
        optimizer = optim.SGD(
            params,
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--base_encoder", type=str, default="resnet18")
        parser.add_argument("--emb_dim", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=32)
        parser.add_argument("--num_negatives", type=int, default=16384)
        parser.add_argument("--encoder_momentum", type=float, default=0.999)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=0.03)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--batch_size", type=int, default=256)
        return parser


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def batch_shuffle_ddp(x):  # pragma: no-cover
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle


@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle):  # pragma: no-cover
    """
    Undo batch shuffle.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]
