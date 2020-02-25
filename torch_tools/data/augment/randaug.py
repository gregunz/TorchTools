from random import random

import PIL.Image
import invertransforms as T

from torch_tools.data.augment.randaug_policies import randaug_policies, NAME_TO_TRANSFORM


class RandAug:

    def __init__(self, dataset='imagenet', norm=True):
        self._dataset_name = dataset
        self.apply_norm = norm

        if self._dataset_name not in self._transforms.keys():
            raise NotImplementedError(f'RandAug for dataset "{self._dataset_name}" not implemented')

        if self._dataset_name == 'imagenet':
            self._all_policies = randaug_policies()
            self._all_policies = [[p1, p2] for p1, p2 in self._all_policies
                                  if p1[0] != p2[0] and
                                  'Rotate' not in (p1[0], p2[0]) and
                                  'Equalize' not in (p1[0], p2[0]) and
                                  'Equalize' not in (p1[0], p2[0]) and
                                  'Cutout' not in (p1[0], p2[0])]

    def __call__(self, img, to_tensor=True, norm=True):
        """

        Args:
            img: PIL Image
            norm: bool

        Returns:

        """
        norm = norm and self.apply_norm
        return self._transforms[self._dataset_name](img, to_tensor=to_tensor, norm=norm)

    @property
    def _transforms(self):
        return {
            'imagenet': self.__randaug_imagenet,
            'mnist': self.__randaug_mnist,
        }

    def __randaug_mnist(self, img_pil, to_tensor, norm):
        return T.Compose([
            T.RandomAffine(
                degrees=10,
                translate=(0.17, 0.17),
                scale=(0.85, 1.05),
                shear=(-10, 10, -10, 10),
                resample=PIL.Image.BILINEAR,
            ),
            T.ColorJitter(0.5, 0.5, 0.5, 0.25),
            T.TransformIf(T.ToTensor(), to_tensor),
            T.TransformIf(T.Normalize(mean=(0.1307,), std=(0.3081,)), to_tensor and norm),
        ])(img_pil)

    def __randaug_imagenet(self, img_pil, to_tensor, norm):
        # TODO: turn this code less cryptic...

        policy = [('FlipLR', 0.5, random.randint(1, 9)),
                  ('FlipUD', 0.5, random.randint(1, 9)),
                  ('Rotate', 0.5, random.randint(1, 9))] \
                 + random.choice(self._all_policies)

        img_shape = img_pil.size[::-1] + (3,)

        for xform in policy:
            assert len(xform) == 3
            name, probability, level = xform
            xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(probability, level, img_shape)
            img_pil = xform_fn(img_pil)

        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.TransformIf(T.ToTensor(), to_tensor),
            T.TransformIf(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), to_tensor and norm),
        ])(img_pil)

    def __to_pil(self, img):
        if not isinstance(img, PIL.Image.Image):
            return T.ToPILImage()(img)
        return img
