import torch
import torch.nn as nn
import torch.nn.functional as F

affine_par = True

class InterpolateNearest2d(nn.Module):
    """
    Custom implementation of nn.Upsample because pytorch/xla
    does not yet support scale_factor and needs to be provided with
    the output_size
    """

    def __init__(self, scale_factor=2):
        """
        Create an InterpolateNearest2d module
        Args:
            scale_factor (int, optional): Output size multiplier. Defaults to 2.
        """
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Interpolate x in "nearest" mode on its last 2 dimensions
        Args:
            x (torch.Tensor): input to interpolate
        Returns:
            torch.Tensor: upsampled tensor with shape
                (...x.shape, x.shape[-2] * scale_factor, x.shape[-1] * scale_factor)
        """
        return F.interpolate(
            x,
            size=(x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor),
            mode="nearest",
        )

class _ASPPModule(nn.Module):
    # https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/aspp.py
    def __init__(
        self, inplanes, planes, kernel_size, padding, dilation, BatchNorm, no_init
    ):
        super().__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        if not no_init:
            self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    # https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/aspp.py
    def __init__(self, backbone, output_stride, BatchNorm, no_init):
        super().__init__()

        if backbone == "mobilenet":
            inplanes = 320
        else:
            inplanes = 2048

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(
            inplanes,
            256,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm,
            no_init=no_init,
        )
        self.aspp2 = _ASPPModule(
            inplanes,
            256,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
            no_init=no_init,
        )
        self.aspp3 = _ASPPModule(
            inplanes,
            256,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
            no_init=no_init,
        )
        self.aspp4 = _ASPPModule(
            inplanes,
            256,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
            no_init=no_init,
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        if not no_init:
            self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabV2Decoder(nn.Module):
    # https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/decoder.py
    # https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/deeplab.py
    def __init__(self, opts, no_init=False):
        super().__init__()
        self.aspp = ASPP("resnet", 16, nn.BatchNorm2d, no_init)

        conv_modules = [
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        ]
        #if opts.gen.s.upsample_featuremaps:
        #    conv_modules = [InterpolateNearest2d(scale_factor=2)] + conv_modules
        output_dim  = 34 #number of classes 
        conv_modules += [
            nn.Conv2d(256, output_dim, kernel_size=1, stride=1),
        ]
        self.conv = nn.Sequential(*conv_modules)

        self._target_size = 224
        print(
            "      - {}:  setting target size to {}".format(
                self.__class__.__name__, self._target_size
            )
        )

    def set_target_size(self, size):
        """
        Set final interpolation's target size
        Args:
            size (int, list, tuple): target size (h, w). If int, target will be (i, i)
        """
        if isinstance(size, (list, tuple)):
            self._target_size = size[:2]
        else:
            self._target_size = (size, size)

    def forward(self, z, z_depth=None):
        if self._target_size is None:
            error = "self._target_size should be set with self.set_target_size()"
            error += "to interpolate logits to the target seg map's size"
            raise Exception(error)
        if isinstance(z, (list, tuple)):
            z = z[0]
        if z.shape[1] != 2048:
            raise Exception(
                "Segmentation decoder will only work with 2048 channels for z"
            )

        y = self.aspp(z)
        y = self.conv(y)
        return F.interpolate(y, self._target_size, mode="bilinear", align_corners=True)

class DeeplabV2Encoder(nn.Module):
    def __init__(self, opts, no_init=False, verbose=0):
        """Deeplab architecture encoder"""
        super().__init__()

        self.model = ResNetMulti([3, 4, 23, 3], 0)
        
        #if input is [1,3,224,224] output will be torch.Size([1, 2048, 28, 28])
        
        #if opts.gen.deeplabv2.use_pretrained and not no_init:
        #    saved_state_dict = torch.load(opts.gen.deeplabv2.pretrained_model)
        #    new_params = self.model.state_dict().copy()
        #    for i in saved_state_dict:
        ##        i_parts = i.split(".")
        #        if not i_parts[1] in ["layer5", "resblock"]:
        #            new_params[".".join(i_parts[1:])] = saved_state_dict[i]
        #    self.model.load_state_dict(new_params)
        #if verbose > 0:
        #        print("    - Loaded pretrained weights")

    def forward(self, x):
        return self.model(x)
    


class BaseDecoder(nn.Module):
    def __init__(self, input_size, target_size):
        super().__init__()
        self.target_size = target_size
        modules = [nn.AdaptiveAvgPool2d((1,1)), 
                   nn.Flatten(), 
                   nn.Linear(input_size,target_size)
            #nn.Linear(2048, 4096),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            #nn.Linear(4096, 8192),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            #nn.Linear(8192,target_size)# 16384),
           # nn.ReLU(),
           # nn.Dropout(0.5),
            #nn.Linear(8192, target_size),
        ]
        #if opts.gen.s.upsample_featuremaps:
        #    conv_modules = [InterpolateNearest2d(scale_factor=2)] + conv_modules

        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.model(x)
        
        
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        # change
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        # change
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=padding,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNetMulti(nn.Module):
    def __init__(
        self,
        layers,
        n_res=4,
        res_norm="instance",
        activ="lrelu",
        pad_type="reflect",
    ):
        self.inplanes = 64
        block = Bottleneck
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, ceil_mode=True
        )  # changed padding from 1 to 0
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.layer_res = ResBlocks(
            n_res, 2048, norm=res_norm, activation=activ, pad_type=pad_type
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (
            stride != 1
            or self.inplanes != planes * block.expansion
            or dilation == 2
            or dilation == 4
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par),
            )
        for i in downsample._modules["1"].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, dilation=dilation, downsample=downsample
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer_res(x)
        return x
    
class ResBlocks(nn.Module):
    """
    From https://github.com/NVlabs/MUNIT/blob/master/networks.py
    """

    def __init__(self, num_blocks, dim, norm="in", activation="relu", pad_type="zero"):
        super().__init__()
        self.model = nn.Sequential(
            *[
                ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return strings.resblocks(self)


class ResBlock(nn.Module):
    def __init__(self, dim, norm="in", activation="relu", pad_type="zero"):
        super().__init__()
        self.dim = dim
        self.norm = norm
        self.activation = activation
        model = []
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type
            )
        ]
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation="none", pad_type=pad_type
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

    def __str__(self):
        return strings.resblock(self)


    # -----------------------------------------
# -----  Generic Convolutional Block  -----
# -----------------------------------------
class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        norm="none",
        activation="relu",
        pad_type="zero",
        bias=True,
    ):
        super().__init__()
        self.use_bias = bias
        # initialize padding
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        use_spectral_norm = False
        if norm.startswith("spectral_"):
            norm = norm.replace("spectral_", "")
            use_spectral_norm = True

        norm_dim = output_dim
        if norm == "batch":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "instance":
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "layer":
            self.norm = LayerNorm(norm_dim)
        elif norm == "adain":
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == "spectral" or norm.startswith("spectral_"):
            self.norm = None  # dealt with later in the code
        elif norm == "none":
            self.norm = None
        else:
            raise ValueError("Unsupported normalization: {}".format(norm))

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=False)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            raise ValueError("Unsupported activation: {}".format(activation))

        # initialize convolution
        if norm == "spectral" or use_spectral_norm:
            self.conv = SpectralNorm(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size,
                    stride,
                    dilation=dilation,
                    bias=self.use_bias,
                )
            )
        else:
            self.conv = nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size,
                stride,
                dilation=dilation,
                bias=self.use_bias if norm != "batch" else False,
            )

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def __str__(self):
        return strings.conv2dblock(self)

