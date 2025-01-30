from torch import nn
import torch
import numpy as np
import warnings
from collections import OrderedDict

from ._abs_layer import AbsLayer


class ChirpLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=15,
        alpha_0=10,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.aplha_0 = alpha_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        x = self.linear(input)
        return torch.sin(self.omega_0 * x + self.aplha_0 * x * torch.tanh(x))

    def forward_with_intermediate(self, input):
        x = self.linear(input)
        # For visualization of activation distributions
        intermediate = self.omega_0 * x + self.aplha_0 * x * torch.tanh(x)
        return torch.sin(intermediate), intermediate


class Chirp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        outermost_activation=AbsLayer(),
        first_omega_0=30,
        hidden_omega_0=30.0,
        first_alpha_0=10,
        hidden_alpha_0=10,
    ):
        super().__init__()
        self.in_features = in_features
        self.net = []
        self.net.append(
            ChirpLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
                alpha_0=first_alpha_0,
            )
        )
        for i in range(hidden_layers):
            self.net.append(
                ChirpLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    alpha_0=hidden_alpha_0,
                )
            )
        if outermost_linear:
            warnings.warn(
                "With outermost_linear=True Chirp network's last layer initialization is based on SIREN's last layer initialization."
                "Most probably also alpha_0 should be used in the initialization, but it is not implemented yet."
            )
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                # FIXME: This is probably not correct, as it should be based on alpha_0 as well.
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
            self.net.append(outermost_activation)
        else:
            self.net.append(
                ChirpLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    alpha_0=hidden_alpha_0,
                )
            )
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        # We must force the putput to be positive as it represents a density.
        output = self.net(coords)
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        """Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!"""
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations["input"] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, ChirpLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations[
                    "_".join((str(layer.__class__), "%d" % activation_count))
                ] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
