from abc import abstractmethod

from .. import Strategy


class GANStrategy(Strategy):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self._num_gen_opt = None
        self._num_dis_opt = None

    def optim_schedulers(self):
        gen_opts, gen_scheds = self.opt_sched_unpack(self.generator_optim_schedulers())
        dis_opts, dis_scheds = self.opt_sched_unpack(self.discriminator_optim_schedulers())
        self._num_gen_opt = len(gen_opts)
        self._num_dis_opt = len(dis_opts)
        return gen_opts + dis_opts, gen_scheds + dis_scheds

    @abstractmethod
    def generator_optim_schedulers(self):
        raise NotImplementedError

    @abstractmethod
    def discriminator_optim_schedulers(self):
        raise NotImplementedError

    def tng_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_idx: int) -> dict:
        self.__check_optimizer_idx(optimizer_idx)
        if optimizer_idx < self._num_gen_opt:
            return self.tng_generator_step(batch, batch_idx, optimizer_idx, epoch_idx)
        else:
            return self.tng_discriminator_step(batch, batch_idx, optimizer_idx - self._num_gen_opt, epoch_idx)

    @abstractmethod
    def tng_generator_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_index: int) -> dict:
        raise NotImplementedError

    @abstractmethod
    def tng_discriminator_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_index: int) -> dict:
        raise NotImplementedError

    def val_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_idx: int) -> dict:
        self.__check_optimizer_idx(optimizer_idx)
        if optimizer_idx < self._num_gen_opt:
            return self.val_generator_step(batch, batch_idx, optimizer_idx, epoch_idx)
        else:
            return self.val_discriminator_step(batch, batch_idx, optimizer_idx - self._num_gen_opt, epoch_idx)

    def val_generator_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_index: int) -> dict:
        pass

    def val_discriminator_step(self, batch, batch_idx: int, optimizer_idx: int, epoch_index: int) -> dict:
        pass

    def tst_step(self, batch, batch_idx: int, optimizer_idx: int) -> dict:
        self.__check_optimizer_idx(optimizer_idx)
        if optimizer_idx < self._num_gen_opt:
            return self.tst_generator_step(batch, batch_idx, optimizer_idx)
        else:
            return self.tst_discriminator_step(batch, batch_idx, optimizer_idx - self._num_gen_opt)

    def tst_generator_step(self, batch, batch_idx: int, optimizer_idx: int) -> dict:
        pass

    def tst_discriminator_step(self, batch, batch_idx: int, optimizer_idx: int) -> dict:
        pass

    def __check_optimizer_idx(self, optimizer_idx):
        if optimizer_idx >= self._num_gen_opt + self._num_dis_opt:
            raise Exception(f'Unexpected optimizer index: {optimizer_idx}, but only received {self._num_gen_opt} '
                            f'optimizers for generator and {self._num_dis_opt} for discriminator.')
