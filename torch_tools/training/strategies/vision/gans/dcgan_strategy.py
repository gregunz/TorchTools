from argparse import ArgumentParser

import torch
from torch import optim, nn
from torch.nn import functional as F

from torch_tools.training.strategies import GANStrategy
from torch_tools.training.util import ImageLogger


class DCGANStrategy(GANStrategy, ImageLogger):
    def __init__(self, generator: nn.Module, discriminator: nn.Module, noise_input_size, lr_gen: float, lr_dis: float,
                 log_dir, output_to_img=None, **kwargs):
        super().__init__(log_dir=log_dir)
        ImageLogger.__init__(self, output_to_image=output_to_img)
        self.generator = generator
        self.discriminator = discriminator
        self.noise_input_size = noise_input_size
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis

        self.lambd = 0.1
        self.generated_imgs = None
        self.log_interval = 20
        self.fixed_noise = torch.randn(32, *self.noise_input_size)  # next(iter(self.tng_dl)).size(0)

    def generator_optim_schedulers(self):
        return optim.Adam(self.generator.parameters(), lr=self.lr_gen, betas=(0.5, 0.999))

    def discriminator_optim_schedulers(self):
        return optim.Adam(self.discriminator.parameters(), lr=self.lr_dis, betas=(0.5, 0.999))

    def tng_generator_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_idx: int, num_batches: int) -> dict:
        x_real, _ = batch

        if epoch_idx == 0 and batch_idx == 0:
            self.log_images(
                tag='training/input',
                images_tensor=x_real.view(4, -1, *x_real.size()[1:]),
                global_step=None,
            )

        if batch_idx == 0:
            images = torch.tanh(self.generator(self.fixed_noise.to(x_real.device)).detach())
            self.log_images(
                tag='training/generated',
                images_tensor=images.view(4, -1, *x_real.size()[1:]),
                global_step=epoch_idx,
            )

        noise = torch.randn(x_real.size(0), *self.noise_input_size).to(x_real.device)
        x_fake = torch.tanh(self.generator(noise))

        logits_fake = self.discriminator(x_fake)
        # we want fake data to be classified as real (real = 1)
        y_fake = torch.ones_like(logits_fake)

        g_loss = F.binary_cross_entropy_with_logits(logits_fake, y_fake)

        self.log(
            metrics_dict={
                'training/batch/generator/loss': g_loss,
            },
            global_step=num_batches * epoch_idx + batch_idx,
            interval=20,
        )

        # passing the the generated image to discriminator
        self.generated_imgs = x_fake.detach()

        return {
            'loss': g_loss,
            'g_loss': g_loss,
        }

    def tng_discriminator_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_idx: int,
                               num_batches: int) -> dict:
        x_real, _ = batch
        logits_real = self.discriminator(x_real)
        y_real = torch.ones_like(logits_real)  # real = 1

        x_fake = self.generated_imgs
        logits_fake = self.discriminator(x_fake)
        y_fake = torch.zeros_like(logits_fake)  # fake = 0

        logits = torch.stack((logits_real, logits_fake))
        targets = torch.stack((y_real, y_fake))

        d_loss = F.binary_cross_entropy_with_logits(logits, targets)

        self.log(
            metrics_dict={
                'training/batch/discriminator/loss': d_loss,
            },
            global_step=num_batches * epoch_idx + batch_idx,
            interval=20,
        )

        return {
            'loss': d_loss,
            'd_loss': d_loss,
        }

    @staticmethod
    def add_argz(parser: ArgumentParser) -> None:
        parser.add_argument('--lr_gen', type=float, default=0.0002)
        parser.add_argument('--lr_dis', type=float, default=0.0002)
