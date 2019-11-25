import io
import warnings
from pathlib import Path
from typing import List, Union

import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, IMAGES, SCALARS


class TensorboardLogs:
    def __init__(self, filepath: Union[Path, str], num_scalars=0, num_images=0):
        size_guidance = {
            SCALARS: num_scalars,
            IMAGES: num_images,
        }
        self._event_acc = EventAccumulator(str(filepath), size_guidance=size_guidance)
        self._event_acc.Reload()
        self.tags = self._event_acc.Tags()
        self.all_tags = [tag for _, some_tags in self.tags.items() if isinstance(some_tags, list) for tag in some_tags]
        if len(self.all_tags) == 0:
            warnings.warn('No tags detected, the path might not contain any logs')

    def images(self, tag: str) -> List[Image.Image]:
        return [Image.open(io.BytesIO(img_event.encoded_image_string)) for img_event in self._event_acc.Images(tag)]

    def scalars(self, tag: str) -> torch.Tensor:
        return torch.tensor([scalar_event.value for scalar_event in self._event_acc.Scalars(tag)])

    # def texts(self, tag: str) -> torch.Tensor:
    #     return torch.tensor([scalar_event.value for scalar_event in self.event_acc.Tags(tag)])
