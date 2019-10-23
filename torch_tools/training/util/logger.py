from abc import abstractmethod, ABCMeta

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class Logger:
    @property
    @abstractmethod
    def logger(self) -> SummaryWriter:
        raise NotImplementedError

    def log_auc_roc(self, tag, outputs, targets, global_step):
        assert isinstance(outputs, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        outputs = outputs.squeeze()
        targets = targets.squeeze()
        assert outputs.dim() == 1
        assert targets.dim() == 1
        assert outputs.size() == targets.size()
        target_values = targets.unique()
        assert target_values.size(0) == 2
        assert set(target_values.tolist()) == {0, 1}

        targets = targets.detach().cpu()
        outputs = outputs.detach().cpu()

        fpr, tpr, threshold = metrics.roc_curve(targets, outputs)

        roc_figure = plt.figure(figsize=(5, 5))
        plt.title('Receiver Operating Characteristic')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.plot([0, 1], [0, 1], 'r--')
        plt.plot(fpr, tpr)

        self.logger.add_figure(
            tag=f'{tag}/roc',
            figure=roc_figure,
            global_step=global_step,
        )

        auc = np.trapz(tpr, fpr)

        self.logger.add_scalar(
            tag=f'{tag}/auc',
            scalar_value=auc,
            global_step=global_step,
        )
        return auc


class ImageLogger(Logger, metaclass=ABCMeta):
    def __init__(self, output_to_image):
        self.output_to_img = output_to_image

    def log_images(self, tag, images_tensor, global_step):
        if images_tensor.dim() == 3:
            images_tensor = images_tensor.unsqueeze(0)
        num_cols = images_tensor.size(0)
        num_rows = images_tensor.size(1)
        images = []
        for i in range(num_cols):
            images += [self.output_to_img(images_tensor[i, img_idx]) for img_idx in range(num_rows)]
        self.logger.add_image(
            tag=tag,
            img_tensor=make_grid(images, nrow=num_rows),
            global_step=global_step,
        )
