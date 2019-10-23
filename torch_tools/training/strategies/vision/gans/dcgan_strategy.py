from typing import Union, List

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torch_tools.training.strategies import GANStrategy
from torch_tools.training.util import ImageLogger


class DCGANStrategy(GANStrategy, ImageLogger):
    def __init__(self, tng_dataloader: DataLoader, generator: nn.Module, discriminator: nn.Module, noise_input_size,
                 log_dir, output_to_img=None, **kwargs):
        super().__init__(log_dir=log_dir)
        ImageLogger.__init__(self, output_to_image=output_to_img)
        self.generator = generator
        self.discriminator = discriminator
        self.noise_input_size = noise_input_size
        self.tng_dl = tng_dataloader

        self.lambd = 0.1
        self.generated_imgs = None
        self.fixed_noise = torch.randn(16, *self.noise_input_size)

    def tng_data_loader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.tng_dl

    def generator_optim_schedulers(self):
        return optim.Adam(self.generator.parameters(), lr=0.0002)

    def discriminator_optim_schedulers(self):
        return optim.Adam(self.generator.parameters(), lr=0.0002)

    def tng_generator_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_idx: int) -> dict:
        x_real, _ = batch

        if batch_idx == 0:
            self.log_images(
                tag='validation/in_out',
                images_tensor=self.generator(self.fixed_noise.to(x_real.device)).view(4, 4, *x_real.size()[1:]),
                global_step=epoch_idx,
            )

        noise = torch.randn(x_real.size(0), *self.noise_input_size).to(x_real.device)
        x_fake = self.generator(noise)
        self.generated_imgs = x_fake.detach()

        logits_fake, _ = self.discriminator(x_fake)
        y_fake = torch.ones_like(logits_fake)

        g_loss = F.binary_cross_entropy(logits_fake, y_fake)

        self.log({
            'training/batch/generator/loss': g_loss,
        }, global_step=self.num_tng_batch * epoch_idx + batch_idx)

        return {
            'loss': g_loss,
        }

    def tng_discriminator_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_idx: int) -> dict:
        x_real, _ = batch
        x_fake = self.generated_imgs

        logits_real, _ = self.discriminator(x_real)
        logits_fake, _ = self.discriminator(x_fake)

        y_real = torch.zeros_like(logits_real)
        y_fake = torch.ones_like(logits_fake)

        real_loss = F.binary_cross_entropy_with_logits(logits_real, y_real)
        fake_loss = F.binary_cross_entropy_with_logits(logits_fake, y_fake)

        d_loss = 0.5 * (real_loss + fake_loss)

        self.log({
            'training/batch/discriminator/loss': d_loss,
        }, global_step=self.num_tng_batch * epoch_idx + batch_idx)

        return {
            'loss': d_loss,
        }
