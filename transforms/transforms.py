
from typing import Dict
from math import ceil
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module  # type: ignore[attr-defined]
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torchvision import transforms as trsfs
# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"

transformable = [ "rgb", "near_ir", "altitude", "landcover"]

class RandomHorizontalFlip:  # type: ignore[misc,name-defined]
    """Horizontally flip the given sample randomly with a given probability."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize a new transform instance.
        Args:
            p: probability of the sample being flipped
        """
        super().__init__()
        self.p = p

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Randomly flip the image and target tensors.
        Args:
            sample: a single data sample
        Returns:
            a possibly flipped sample
        """
        if torch.rand(1) < self.p:
            for s in sample:
                sample[s] = sample[s].flip(-1)

        return sample


class RandomVerticalFlip:  # type: ignore[misc,name-defined]
    """Vertically flip the given sample randomly with a given probability."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize a new transform instance.
        Args:
            p: probability of the sample being flipped
        """
        super().__init__()
        self.p = p

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Randomly flip the image and target tensors.
        Args:
            sample: a single data sample
        Returns:
            a possibly flipped sample
        """

        if torch.rand(1) < self.p:
            for s in sample:
                sample[s] = sample[s].flip(-2)
        return sample


class RandomGaussianNoise:  # type: ignore[misc,name-defined]
    """Identity function used for testing purposes."""

    def __init__(self, max_noise=5e-2, std=1e-2):

        self.max = max_noise
        self.std = std

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            sample: the input
        Returns:
            theinput with added gaussian noise
        """        
        noise = torch.normal(0, self.std, sample["img"].size())
        noise = torch.clamp(sample["img"], min=0, max=self.max)
        sample["img"] += noise
        return sample


class Resize:
    def __init__(self, size):
        """
        size = (height, width) target size 
        """
        self.h, self.w = size

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for s in sample:
            if s in transformable:
                sample[s] = F.interpolate(sample[s].float(), size=(self.h, self.w), mode="nearest")
        return sample

        for s in sample:
            sample[s] = F.interpolate(sample[s].float(), size=(self.h, self.w), mode="nearest")
        
        return sample


class Normalize:
    def __init__(self, band, means, std):
        """
        size = (height, width) target size 
        """
        self.means = means
        self.std = std
        self.band = band

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:   
        sample[self.band] = normalize(sample[self.band], self.means, self.std)
        return sample

    
class RandomCrop:  # type: ignore[misc,name-defined]
    """Identity function used for testing purposes."""
    
    def __init__(self, size, center=False):
        assert isinstance(size, (int, tuple, list))
        if not isinstance(size, int):
            assert len(size) == 2
            self.h, self.w = size
        else:
            self.h = self.w = size

        self.h = int(self.h)
        self.w = int(self.w)
        self.center = center
        
    def  __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            sample: the input
        Returns:
            the cropped input
        """
     
        H, W = (
            sample["rgb"].size()[-2:] if "rgb" in sample else list(sample.values())[0].size()[-2:]
        )
        if not self.center:                
            top = max(0, np.random.randint(0, max(H - self.h,1)))
            left = max(0, np.random.randint(0, max(W - self.w,1)))
        else:
            top = max(0, (H - self.h) // 2)
            left = max(0,(W - self.w) // 2)
        
        return {
            task: tensor[:,:,top : top + self.h, left : left + self.w]
            for task, tensor in sample.items()
        }

    
class DBN(nn.Module):
    def __init__(self, 
                 num_features, 
                 num_groups=32, 
                 num_channels=0, 
                 dim=4, eps=1e-5, 
                 momentum=0.1, 
                 affine=False, 
                 mode=0,
                 *args, **kwargs):
        super(DBN, self).__init__()
        if num_channels > 0:
            num_groups = num_features // num_channels
        self.num_features = num_features
        self.num_groups = num_groups
        assert self.num_features % self.num_groups == 0
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode

        self.shape = [1] * dim
        self.shape[1] = num_features

        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, 1))
        self.register_buffer('running_projection', torch.eye(num_groups))
        self.reset_parameters()

    # def reset_running_stats(self):
    #     self.running_mean.zero_()
    #     self.running_var.eye_(1)

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def __call__(self, input: torch.Tensor):
        size = input["rgb"].size()
        assert input["rgb"].dim() == self.dim and size[1] == self.num_features
        x = input["rgb"].view(size[0] * size[1] // self.num_groups, self.num_groups, *size[2:])
        training = self.mode > 0 or (self.mode == 0 and self.training)
        x = x.transpose(0, 1).contiguous().view(self.num_groups, -1)
        if training:
            mean = x.mean(1, keepdim=True)
#             self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
            x_mean = x - mean
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_groups, device=input["rgb"].device)
            # print('sigma size {}'.format(sigma.size()))
            u, eig, _ = sigma.svd()
            scale = eig.rsqrt()
            wm = u.matmul(scale.diag()).matmul(u.t())
#             self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm
            y = wm.matmul(x_mean)
        else:
            x_mean = x - self.running_mean
            y = self.running_projection.matmul(x_mean)
        output = y.view(self.num_groups, size[0] * size[1] // self.num_groups, *size[2:]).transpose(0, 1)
        output = output.contiguous().view_as(input["rgb"])
        if self.affine:
            output = output * self.weight + self.bias
            
        input["rgb"] = output
        return input

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)
    
    
def get_transform(transform_item, mode):
    """Returns the transform function associated to a
    transform_item listed in opts.data.transforms ; transform_item is
    an addict.Dict
    """

    if transform_item.name == "hflip" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandomHorizontalFlip(p=transform_item.p or 0.5)

    elif transform_item.name == "vflip" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandomVerticalFlip(p=transform_item.p or 0.5)
    
    elif transform_item.name == "randomnoise" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandomGaussianNoise(max_noise = transform_item.max_noise or 5e-2, std = transform_item.std or 1e-2)
    
    elif transform_item.name == "normalize" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):

        return Normalize(transform_item.band, transform_item.means, transform_item.std)
    
    elif transform_item.name == "resize" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        
        return Resize(size=transform_item.size)
    
    elif transform_item.name == "crop" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandomCrop(transform_item.size, transform_item.center)
    
    elif transform_item.name == "whiten" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return DBN(transform_item.num_feats, 
                   num_groups=transform_item.num_groups, 
                  momentum=transform_item.momentum)
    

    elif transform_item.ignore is True or transform_item.ignore == mode:
        return None

    raise ValueError("Unknown transform_item {}".format(transform_item))

    
def get_transforms(opts, mode):
    """Get all the transform functions listed in opts.data.transforms
    using get_transform(transform_item, mode)
    """
    transforms = []

    for t in opts.data.transforms:
        transforms.append(get_transform(t, mode))
    

    transforms = [t for t in transforms if t is not None]

    return trsfs.Compose(transforms)

