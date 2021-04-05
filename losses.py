import torch
import math

class RelativeMSE(torch.nn.Module):
    """Relative Mean-Squared Error.
    :math:`0.5 * \\frac{(x - y)^2}{y^2 + \epsilon}`
    Args:
        eps(float): small number to avoid division by 0.
    """
    def __init__(self, eps=1e-2):
        super(RelativeMSE, self).__init__()
        self.eps = eps

    def forward(self, im, ref):
        """Evaluate the metric.
        Args:
            im(torch.Tensor): image.
            ref(torch.Tensor): reference.
        """
        mse = torch.pow(im-ref, 2)
        loss = mse/(torch.pow(ref, 2) + self.eps)
        loss = 0.5*torch.mean(loss)
        return loss

class SMAPE(torch.nn.Module):
    """Symmetric Mean Absolute error.
    :math:`\\frac{|x - y|} {|x| + |y| + \epsilon}`
    Args:
        eps(float): small number to avoid division by 0.
    """

    def __init__(self, eps=1e-2):
        super(SMAPE, self).__init__()
        self.eps = eps

    def forward(self, im, ref):
        # NOTE: the denominator is used to scale the loss, but does not
        # contribute gradients, hence the '.detach()' call.
        loss = (torch.abs(im-ref) / (
            self.eps + torch.abs(im.detach()) + torch.abs(ref.detach()))).mean()

        return loss

class TonemappedMSE(torch.nn.Module):
    """Mean-squared error on tonemaped images.
    Args:
        eps(float): small number to avoid division by 0.
    """

    def __init__(self, eps=1e-2):
        super(TonemappedMSE, self).__init__()
        self.eps = eps  # avoid division by zero

    def forward(self, im, ref):
        im = _tonemap(im)
        ref = _tonemap(ref)
        loss = torch.pow(im-ref, 2)
        loss = 0.5*torch.mean(loss)
        return loss


class TonemappedRelativeMSE(torch.nn.Module):
    """Relative mean-squared error on tonemaped images.
    Args:
        eps(float): small number to avoid division by 0.
    """
    def __init__(self, eps=1e-2):
        super(TonemappedRelativeMSE, self).__init__()
        self.eps = eps  # avoid division by zero

    def forward(self, im, ref):
        im = _tonemap(im)
        ref = _tonemap(ref)
        mse = torch.pow(im-ref, 2)
        loss = mse/(torch.pow(ref, 2) + self.eps)
        loss = 0.5*torch.mean(loss)
        return loss


def _tonemap(im):
    """Helper Reinhards tonemapper.
    Args:
        im(torch.Tensor): image to tonemap.
    Returns:
        (torch.Tensor) tonemaped image.
    """
    im = torch.clamp(im, min=0)
    return im / (1+im)