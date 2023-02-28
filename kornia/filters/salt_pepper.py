from torch import Tensor, clone, ceil, randint
from typing import Union

def salt_pepper(image: Tensor, pixel_noise_p: Union[float, Tensor]) -> Tensor:
    r"""Adjust color saturation of an image.

    .. image:: _static/img/adjust_saturation.png

    The image is expected to be an RGB image in the range of [0, 1].

    Args:
        image: Image/Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
        factor: How much to adjust the saturation. 0 will give a black
          and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.
        saturation_mode: The mode to adjust saturation.

    Return:
        Adjusted image in the shape of :math:`(*, 3, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       image_enhancement.html>`__.

    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> adjust_saturation(x, 2.).shape
        torch.Size([1, 3, 3, 3])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.tensor([1., 2.])
        >>> adjust_saturation(x, y).shape
        torch.Size([2, 3, 3, 3])
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")
    
    channels = image.size(dim=1)
    out = clone(image)
    # Half of the noise for salt and half for pepper.
    num_salt = ceil(0.5 * image.size(dim=2)* image.size(dim=3) * pixel_noise_p) 
    coords = [randint(0, i - 1, int(num_salt))
            for i in image.size()[1:]]
    out[coords] = 1

    # Pepper mode
    num_pepper = ceil(0.5 * image.size(dim=2)* image.size(dim=3) * pixel_noise_p) 
    coords = [randint(0, i - 1, int(num_pepper))
            for i in image.size()[1:]]
    out[coords] = 0

    return out