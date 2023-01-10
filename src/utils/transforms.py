import numpy as np
import albumentations as A
import torchvision


class AlbumentationsTransforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms
        # Torch expect grayscale image to be single dimension, but augmentations returns 3 channel image
        self._to_gray = A.ToGray in [type(t) for t in transforms.transforms]

    # You have to pass data to augmentations as named arguments
    # When using albumentations
    def __call__(self, img, *args, **kwargs):
        img = self.transforms(image=np.array(img))["image"]
        if self._to_gray:
            img = torchvision.transforms.Grayscale()(img)
        return img