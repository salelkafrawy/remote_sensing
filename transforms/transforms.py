
from typing import Dict
from math import ceil
import numpy as np
import torch
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

