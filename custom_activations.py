import torch
from torch import nn


class SoftExp(nn.Module):
    """
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, alpha=None, beta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = torch.device("cpu")
        alpha = torch.tensor(alpha) if alpha is not None else nn.Parameter(torch.randn(1)) # noqa
        self.alpha = alpha.to(self.device)
        self.alpha.requiresGrad = True  # set requiresGrad to true
        # self.__name__ == "SoftExp"

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        if self.alpha == 0.0:
            return x

        if self.alpha < 0.0:
            return -torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        if self.alpha > 0.0:
            return (torch.exp(self.alpha * x) - 1) / self.alpha + self.alpha


@torch.jit.script
def sech(x):
    return 1 / torch.cosh(x)


@torch.jit.script
def dip(x):
    return (-2.0261193218831233 * sech(x)) + 0.31303528549933146


@torch.jit.script
def bmu(x):
    return torch.where(
        x <= -1,
        -1 / torch.abs(x),
        torch.where(x >= 1, x - 2, dip(x)),
    )


class BMU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return bmu(input)


class TrainableHybrid(nn.Module):
    def __init__(
        self, functions, function_args=None, function_kwargs=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if function_args is None:
            function_args = [tuple() for _ in functions]
        if function_kwargs is None:
            function_kwargs = [dict() for _ in functions]
        if None in function_args:
            function_args = [
                tuple() if fa is None else fa for fa in function_args
            ]
        if None in function_kwargs:
            function_kwargs = [
                dict() if fk is None else fk for fk in function_kwargs
            ]
        self.functions = [
            f(*fa, *fk) for f, fa, fk in zip(functions, function_args, function_kwargs)
        ]
        self.alpha = nn.Parameter(torch.randn(len(functions)))
        self.normalize_alpha()
        self.__name__ = (
            f"TrainableHybrid{str([f.__name__ for f in functions]).replace(' ', '')}"
        )
    
    def __repr__(self):
        return self.__name__

    def normalize_alpha(self) -> None:
        self.alpha.data = self.alpha / torch.sum(self.alpha)

    def apply_activations(self, input: torch.Tensor):
        return torch.sum(
            torch.stack(
                [a * f(input) for f, a in zip(self.functions, self.alpha)]
            ),
            dim=0,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.normalize_alpha()
        return self.apply_activations(input)

    def to(self, device):
        super().to(device)
        self.functions = [f.to(device) for f in self.functions]
        return self


class ISRU(nn.Module):
    def __init__(self, alpha=None, beta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = torch.device("cpu")
        alpha = torch.tensor(alpha) if alpha is not None else nn.Parameter(torch.randn(1)) # noqa
        self.alpha = alpha.to(self.device)
        self.alpha.requiresGrad = True
        self.__name__ = "ISRU"

    def forward(self, x):
        return x / torch.sqrt(1 + self.alpha * x**2)


class ISRLU(nn.Module):
    def __init__(self, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = torch.device("cpu")
        alpha = torch.tensor(alpha) if alpha is not None else nn.Parameter(torch.randn(1)) # noqa
        self.alpha = alpha.to(self.device)
        self.alpha.requiresGrad = True
        self.isru = ISRU(alpha)
        self.__name__ = "ISRLU"

    def forward(self, x):
        return torch.where(x >= 0, x, self.isru(x))


class PBessel(nn.Module):
    def __init__(self, alpha=None, beta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = torch.device("cpu")
        alpha = torch.tensor(alpha) if alpha is not None else nn.Parameter(torch.randn(1)) # noqa
        beta = torch.tensor(beta) if beta is not None else nn.Parameter(torch.randn(1))
        self.alpha = alpha.to(self.device)
        self.beta = beta.to(self.device)
        self.alpha.requiresGrad = True
        self.beta.requiresGrad = True
        self.__name__ = "PBessel"

    def forward(self, input):
        gamma = 1 - self.alpha
        return (self.alpha * torch.special.bessel_j0(self.beta * input)) + (
            gamma * torch.special.bessel_j1(self.beta * input)
        )


class LeakyPReQU(nn.Module):
    def __init__(self, alpha=None, beta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = torch.device("cpu")
        alpha = torch.tensor(alpha) if alpha is not None else nn.Parameter(torch.randn(1)) # noqa
        beta = torch.tensor(beta) if beta is not None else nn.Parameter(torch.randn(1))
        self.alpha = alpha.to(self.device)
        self.beta = beta.to(self.device)
        self.alpha.requiresGrad = True
        self.beta.requiresGrad = True
        self.__name__ = "LeakyPReQU"

    def forward(self, input):
        return torch.where(
            input > 0,
            (self.alpha * input * input) + (self.beta * input),
            self.beta * input,
        )


class Sinusoid(nn.Module):
    def __init__(self, alpha=None, beta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = torch.device("cpu")
        alpha = torch.tensor(alpha) if alpha is not None else nn.Parameter(torch.randn(1)) # noqa
        beta = torch.tensor(beta) if beta is not None else nn.Parameter(torch.randn(1))
        self.alpha = alpha.to(self.device)
        self.beta = beta.to(self.device)
        self.alpha.requiresGrad = True
        self.beta.requiresGrad = True
        self.__name__ = "Sinusoid"

    def forward(self, input):
        return torch.sin(self.alpha * (input + self.beta))


class Modulo(nn.Module):
    def __init__(self, alpha=None, beta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = torch.device("cpu")
        alpha = torch.tensor(alpha) if alpha is not None else nn.Parameter(torch.randn(1)) # noqa
        beta = torch.tensor(beta) if beta is not None else nn.Parameter(torch.randn(1))
        self.alpha = alpha.to(self.device)
        self.beta = beta.to(self.device)
        self.alpha.requiresGrad = True
        self.beta.requiresGrad = True
        self.__name__ = "Modulo"

    def forward(self, input):
        return torch.fmod(self.alpha * input, self.beta)


class TriWave(nn.Module):
    def __init__(self, alpha=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = torch.device("cpu")
        alpha = torch.tensor(alpha) if alpha is not None else nn.Parameter(torch.randn(1)) # noqa
        self.alpha = alpha.to(self.device)
        self.alpha.requiresGrad = True
        self.__name__ = "TriWave"

    def forward(self, input):
        return torch.abs(2 * (input / self.alpha - torch.floor(input / self.alpha + 0.5))) # noqa


class Gaussian(nn.Module):
    def __init__(self, alpha=None, beta=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = torch.device("cpu")
        alpha = torch.tensor(alpha) if alpha is not None else nn.Parameter(torch.randn(1)) # noqa
        beta = torch.tensor(beta) if beta is not None else nn.Parameter(torch.randn(1))
        self.alpha = alpha.to(self.device)
        self.beta = beta.to(self.device)
        self.alpha.requiresGrad = True
        self.beta.requiresGrad = True
        self.__name__ = "Gaussian"

    def forward(self, x):
        return torch.exp(-(((x-self.alpha)**2)/(2*self.beta**2)))