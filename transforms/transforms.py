from typing import Dict
from math import ceil
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module  # type: ignore[attr-defined]
from torchvision.transforms.functional import normalize
import torch.nn.functional as F

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


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
            sample = sample.flip(-1)
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

        noise = torch.normal(0, self.std, sample.size())
        noise = torch.clamp(sample, min=0, max=self.max)
        sample += noise
        return sample


class Resize:
    def __init__(self, size):
        """
        size = (height, width) target size 
        """
        self.h, self.w = size

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:

        sample = F.interpolate(sample[s].float(), size=(self.h, self.w), mode="nearest")
        return sample

