import torch
import numpy as np


class directional_encoding:
    """ Directional encoding
    x = [x,y,z] is encoded as [ix, iy, iz, r]
    """

    def __init__(self):
        self.dim = 4
        self.name = "directional_encoding"
    # sp: sampled points

    def __call__(self, sp):
        unit = sp / torch.norm(sp, dim=1).view(-1, 1)
        return torch.cat((unit, torch.norm(sp, dim=1).view(-1, 1)), dim=1)


class positional_encoding:
    """ Positional encoding
    x = [x,y,z] is encoded as [sin(pi x), sin(pi y), sin(pi z), cos(pi x), cos(pi y), cos(pi z), sin(2 pi x), ....]
    """

    def __init__(self, N):
        self.dim = 6 * N
        self.name = "positional_encoding_" + str(N)

    def __call__(self, sp):
        retval = torch.cat((torch.sin(np.pi * sp).view(-1, 3),
                            torch.cos(np.pi * sp).view(-1, 3)), dim=1)
        for i in range(1, self.dim // 6):
            retval = torch.cat((retval, torch.sin(
                2**i*np.pi * sp).view(-1, 3), torch.cos(2**i*np.pi * sp).view(-1, 3)), dim=1)
        return retval


class direct_encoding:
    """Direct encoding:
    x = [x,y,z] is encoded as [x,y,z]
    """

    def __init__(self):
        self.dim = 3
        self.name = "direct_encoding"

    def __call__(self, sp):
        return sp


class spherical_coordinates:
    """Spherical encoding:
    x = [x,y,z] is encoded as [r,phi,theta] (i.e. spherical coordinates)
    """

    def __init__(self):
        self.dim = 3
        self.name = "spherical_coordinates"

    def __call__(self, sp):
        phi = torch.atan2(sp[:, 1], sp[:, 0]) / np.pi
        r = torch.norm(sp, dim=1)
        theta = torch.div(sp[:, 2], r)
        return torch.cat((r.view(-1, 1), phi.view(-1, 1), theta.view(-1, 1)), dim=1)
