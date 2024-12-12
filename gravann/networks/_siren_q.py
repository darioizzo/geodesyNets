# From https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
from torch import nn
import torch
import numpy as np
import math

from ._abs_layer import AbsLayer

class Quadratic(nn.Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight_r: torch.Tensor
    weight_g: torch.Tensor
    weight_b: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_r = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.weight_g = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.weight_b = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias_r = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_g = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self.c = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias_r", None)
            self.register_parameter("bias_g", None)
            self.register_parameter("c", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_b, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_g, a=math.sqrt(5))
        if self.bias_r is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_r)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_r, -bound, bound)
            nn.init.uniform_(self.bias_g, -bound, bound)
            nn.init.uniform_(self.c, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r = nn.functional.linear(input, self.weight_r, self.bias_r)
        g = nn.functional.linear(input, self.weight_g, self.bias_g)
        b = nn.functional.linear(input**2, self.weight_b, self.c)
        return r * g + b

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class SineQLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.quadratic = Quadratic(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.quadratic.weight_r.uniform_(
                    -1 / self.in_features,
                    1 / self.in_features
                )
            else:
                self.quadratic.weight_r.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )
            self.quadratic.weight_g.normal_(mean=0, std=1e-16)
            self.quadratic.weight_b.normal_(mean=0, std=1e-16)
            self.quadratic.bias_g.fill_(1)
            self.quadratic.c.fill_(0)
            

    def forward(self, input):
        return torch.sin(self.omega_0 * self.quadratic(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.quadratic(input)
        return torch.sin(intermediate), intermediate


class SirenQ(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, outermost_activation=AbsLayer(),
                 first_omega_0=30, hidden_omega_0=30.):

        super().__init__()

        self.in_features = in_features
        self.net = []
        self.net.append(SineQLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineQLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = Quadratic(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight_r.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0
                )
                final_linear.weight_g.normal_(mean=0, std=1e-16)
                final_linear.weight_b.normal_(mean=0, std=1e-16)
                final_linear.bias_g.fill_(1)
                final_linear.c.fill_(0)

            self.net.append(final_linear)
            self.net.append(outermost_activation)
        else:
            self.net.append(SineQLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(
            True)  # allows to take derivative w.r.t. input
        # We must force the putput to be positive as it represents a density.
        output = self.net(coords)
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineQLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join(
                    (str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join(
                (str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
