from abc import abstractmethod, ABCMeta

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from torch_tools.metrics import compute_roc_auc, create_roc_figure


class Logger:
    @property
    @abstractmethod
    def logger(self) -> SummaryWriter:
        raise NotImplementedError

    def log_auc_roc(self, tag, outputs, targets, global_step):
        fpr, tpr, _, auc = compute_roc_auc(outputs, targets)
        roc_figure = create_roc_figure(fpr, tpr)

        self.logger.add_figure(
            tag=f'{tag}/roc',
            figure=roc_figure,
            global_step=global_step,
        )

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
