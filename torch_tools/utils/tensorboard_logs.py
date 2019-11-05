import io
from typing import List

import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, IMAGES, SCALARS


class TensorboardLogs:
    def __init__(self, filepath, num_scalars=0, num_images=0):
        size_guidance = {
            SCALARS: num_scalars,
            IMAGES: num_images,
        }
        self.event_acc = EventAccumulator(str(filepath), size_guidance=size_guidance)
        self.event_acc.Reload()

    def images(self, tag: str) -> List[Image.Image]:
        return [Image.open(io.BytesIO(img_event.encoded_image_string)) for img_event in self.event_acc.Images(tag)]

    def scalars(self, tag: str) -> torch.Tensor:
        return torch.tensor([scalar_event.value for scalar_event in self.event_acc.Scalars(tag)])

    # def texts(self, tag: str) -> torch.Tensor:
    #     return torch.tensor([scalar_event.value for scalar_event in self.event_acc.Tags(tag)])
