from torch import nn

from torch_tools.metrics import RunningMoments


class AdaptiveBN(nn.Module):
    def __init__(self, bn, force_bn_eval=False):
        super().__init__()

        if force_bn_eval:
            bn = bn.eval()

        self.bn = bn
        self.force_bn_eval = force_bn_eval

        self.rs = RunningMoments()
        self.saved_running_mean = None
        self.saved_running_var = None

    def train(self, mode=True):
        mode_changed = self.training != mode
        super().train(mode)  # this will set the mode of self.bn

        if self.force_bn_eval:
            self.bn.eval()

        if mode:  # training
            if mode_changed and self.saved_running_var is not None:
                self.bn.running_mean = self.saved_running_mean
                self.bn.running_var = self.saved_running_var

        else:  # eval
            if mode_changed:
                self.saved_running_mean = self.bn.running_mean
                self.saved_running_var = self.bn.running_var

                self.bn.running_mean = self.rs.mean
                self.bn.running_var = self.rs.var

        return self

    def forward(self, x):
        if self.training:
            self.rs.push(x.mean(dim=[i for i in range(x.dim()) if i != 1]))

        return self.bn(x)
