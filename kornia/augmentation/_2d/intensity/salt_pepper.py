from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation import random_generator as rg
from kornia.filters import salt_pepper


class RandomSaltPepper(IntensityAugmentationBase2D):
    r"""Apply solt and pepper noise according to a probability p value.

    Args:
        p: probability of the image to be affected by the noise.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :func:`kornia.color.rgb_to_grayscale`.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.randn((1, 3, 3, 3))
        >>> aug = RandomGrayscale(p=1.0)
        >>> aug(inputs)
        tensor([[[[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]],
        <BLANKLINE>
                 [[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]],
        <BLANKLINE>
                 [[-1.1344, -0.1330,  0.1517],
                  [-0.0791,  0.6711, -0.1413],
                  [-0.1717, -0.9023,  0.0819]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomGrayscale(p=1.0)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    def __init__(
        self, 
        pixel_noise_p: Tuple[float, float] = (0.05, 0.05),
        same_on_batch: bool = False,
        p: float = 0.1,
        keepdim: bool = False
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self._param_generator = rg.PlainUniformGenerator((self.pixel_noise_p, "pixel_noise_p", None, None))

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        pixel_noise_p = params["pixel_noise_p"].to(input)
        return salt_pepper(input, pixel_noise_p)