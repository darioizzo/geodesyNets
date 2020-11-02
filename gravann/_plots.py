from ._mesh_conversion import create_mesh_from_cloud, create_mesh_from_model
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import torch
import math
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
pv.set_plot_theme("night")


def plot_model_vs_cloud_mesh(model, gt_mesh, encoding, save_path=None):
    """Creates a side by side of the model and the ground truth mesh passed to this

    Args:
        model (torch nn): trained model
        gt_mesh (pyvista mesh): ground-truth mesh
        encoding (func): encoding function for the model
        save_path (str, optional): Pass to store plot, if none will display. Defaults to None.
    """
    model_mesh = create_mesh_from_model(
        model, encoding, rho_threshold=1.5e-2, plot_each_it=-1)

    p = pv.Plotter(shape=(1, 2))

    p.subplot(0, 0)
    p.show_grid()
    p.add_text("Model Prediction", font_size=12)
    p.add_mesh(model_mesh, color="grey", show_edges=False, smooth_shading=True)

    p.subplot(0, 1)
    p.show_grid()
    p.add_text("Ground Truth", font_size=12)
    p.add_mesh(gt_mesh, color="grey", show_edges=False, smooth_shading=True)

    if save_path is None:
        p.show()
    else:
        p.save_graphic(save_path, title="")
        p.close()


def plot_points(points):
    """Creates a 3D scatter plot of passed points.     

    Args:
        points (torch tensor): Points to plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0].cpu().numpy(),
               points[:, 1].cpu().numpy(),
               points[:, 2].cpu().numpy())
    plt.show()


def plot_model_mesh(model, encoding, interactive=False, rho_threshold=1.5e-2):
    """Plots the mesh generated from a model that predicts rho. Returns the mesh

    Args:
        model (Torch Model): Model to use 
        encoding (Encoding function): The function used to encode points for the model
        interactive (bool, optional): Creates a separate window which you can use interactively. Defaults to True.
    """
    mesh = create_mesh_from_model(model, encoding, rho_threshold=rho_threshold)
    plot_mesh(mesh, smooth_shading=True,
              show_edges=False, interactive=interactive)
    return mesh


def plot_point_cloud_mesh(cloud, distance_threshold=0.125, use_top_k=1, interactive=False):
    """Display a mesh generated from a point cloud. Returns the mesh

    Args:
        cloud (torch tensor): The points that should be used to generate the mesh (3,N)
        distance_threshold (float): Distance threshold for the mesh generation algorithm. Use larger ones if mesh is broken up into. 
        use_top_k (int): the number of nearest neighbours to be used for distance.
        interactive (bool): Creates a separate window which you can use interactively.
    """
    mesh = create_mesh_from_cloud(cloud.cpu().numpy(
    ), use_top_k=use_top_k, distance_threshold=distance_threshold)
    plot_mesh(mesh, smooth_shading=True,
              show_edges=False, interactive=interactive)
    return mesh


def plot_mesh(mesh, show_edges=True, smooth_shading=False, interactive=True, elev=45, azim=125):
    """Plots a mesh ()

    Args:
        mesh (pyvista mesh): mesh to plot
        show_edges (bool,): Show grid wires. 
        smooth_shading (bool): Use smooth_shading.
        interactive (bool): Creates a separate window which you can use interactively.
    """
    # Plot mesh
    if interactive:
        p = pvqt.BackgroundPlotter()
    else:
        p = pv.Plotter()
    p.show_grid()
    p.add_mesh(mesh, color="grey", show_edges=show_edges,
               smooth_shading=smooth_shading)
    p.show()


def plot_mascon(points, masses=None, elev=45, azim=125, alpha=0.1, s=None):
    """Plots a mascon model

    Args:
        points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points
        masses (1-D array-like): a (N,) array-like object containing the values for the point masses
        elev (float): elevation of the starting 3D view
        azim (float): azimuth for the starting 3D view
        alpha (float): alpha for the mass visualization
        s (int): scale for the visualized masses

    """
    x = points[:, 0].cpu()
    y = points[:, 1].cpu()
    z = points[:, 2].cpu()

    if s is None:
        s = 22000 / len(points)

    if masses is None:
        normalized_masses = s
    else:
        normalized_masses = masses / sum(masses)
        normalized_masses = (normalized_masses * s * len(x)).cpu()

    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')

    # And visualize the masses
    ax.scatter(x, y, z, color='k', s=normalized_masses, alpha=alpha)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=elev, azim=azim)

    ax2 = fig.add_subplot(222)
    ax2.scatter(x, y, color='k', s=normalized_masses, alpha=alpha)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])

    ax3 = fig.add_subplot(223)
    ax3.scatter(x, z, color='k', s=normalized_masses, alpha=alpha)
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])

    ax4 = fig.add_subplot(224)
    ax4.scatter(y, z, color='k', s=normalized_masses, alpha=alpha)
    ax4.set_xlim([-1, 1])
    ax4.set_ylim([-1, 1])

    plt.show()


def plot_model_grid(model, encoding, N=20, bw=False, alpha=0.2, views_2d=True, c=1.):
    """Plots the neural model of the asteroid density in the [-1,1]**3 cube showing
    the density value on a grid.

    Args:
        model (callable (a,b)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        N (int): grid size (N**3 points will be plotted).
        bw (bool): when True considers zero density as white and transparent. The final effect is a black and white plot
        alpha (float): alpha for the visualization
        views_2d (bool): activates also the 2d projections
        c (float, optional): Normalization constant. Defaults to 1.

    """

    # We create the grid
    x = torch.linspace(-1, 1, N)
    y = torch.linspace(-1, 1, N)
    z = torch.linspace(-1, 1, N)
    X, Y, Z = torch.meshgrid((x, y, z))

    # We compute the density on the grid points (no gradient as its only for plotting)
    nn_inputs = torch.cat(
        (X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), dim=1)
    nn_inputs = encoding(nn_inputs)
    RHO = model(nn_inputs).detach()*c

    # And we plot it
    fig = plt.figure()
    if views_2d:
        ax = fig.add_subplot(221, projection='3d')
    else:
        ax = fig.add_subplot(111, projection='3d')
    if bw:
        col = torch.cat((1-RHO, 1-RHO, 1-RHO, RHO), dim=1).cpu()
        alpha = None
    else:
        col = RHO.cpu()

    ax.scatter(X.reshape(-1, 1).cpu(), Y.reshape(-1, 1).cpu(), Z.reshape(-1, 1).cpu(),
               marker='.', c=col, s=100, alpha=alpha)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=45., azim=125.)

    if views_2d:
        ax2 = fig.add_subplot(222)
        ax2.scatter(X.reshape(-1, 1)[:, 0].cpu(), Y.reshape(-1, 1)[:, 0].cpu(),
                    marker='.', c=col, s=100, alpha=alpha)
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])

        ax3 = fig.add_subplot(223)
        ax3.scatter(X.reshape(-1, 1)[:, 0].cpu(), Z.reshape(-1, 1)[:, 0].cpu(),
                    marker='.', c=col, s=100, alpha=alpha)
        ax3.set_xlim([-1, 1])
        ax3.set_ylim([-1, 1])

        ax4 = fig.add_subplot(224)
        ax4.scatter(Y.reshape(-1, 1)[:, 0].cpu(), Z.reshape(-1, 1)[:, 0].cpu(),
                    marker='.', c=col, s=100, alpha=alpha)
        ax4.set_xlim([-1, 1])
        ax4.set_ylim([-1, 1])

    plt.show()


def plot_model_rejection(model, encoding, N=30**3, views_2d=False, bw=False, alpha=0.2, crop_p=1e-2, s=100, save_path=None, c=1.):
    """Plots the neural model of the asteroid density in the [-1,1]**3 cube interpreting the density
    as a probability distribution and performing a rejection sampling approach

    Args:
        model (callable (N,M)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        N (int): number of points to be considered.
        views_2d (bool): activates also the 2d projections
        bw (bool): results in a black and white plot
        alpha (float): alpha for the visualization
        crop_p (float): all points below this density are rejected
        s (int): size of the non rejected points visualization
        save_path (str, optional): Pass to store plot, if none will display. Defaults to None.
        c (float, optional): Normalization constant. Defaults to 1.

    """
    points = torch.rand(N, 3) * 2 - 1
    nn_inputs = encoding(points)
    RHO = model(nn_inputs).detach() * c
    mask = RHO > (torch.rand(N, 1) + crop_p)
    RHO = RHO[mask]
    points = [[it[0].item(), it[1].item(), it[2].item()]
              for it, m in zip(points, mask) if m]
    if len(points) == 0:
        print("All points rejected! Plot is empty, try cropping less?")
        return
    points = torch.tensor(points)
    fig = plt.figure()
    if views_2d:
        ax = fig.add_subplot(221, projection='3d')
    else:
        ax = fig.add_subplot(111, projection='3d')
    if bw:
        col = 'k'
    else:
        col = RHO.cpu()
    # And we plot it
    ax.scatter(points[:, 0].cpu(), points[:, 1].cpu(), points[:, 2].cpu(),
               marker='.', c=col, s=s, alpha=alpha)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=45., azim=125.)

    if views_2d:
        ax2 = fig.add_subplot(222)
        ax2.scatter(points[:, 0].cpu(), points[:, 1].cpu(),
                    marker='.', c=col, s=s, alpha=alpha)
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])

        ax3 = fig.add_subplot(223)
        ax3.scatter(points[:, 0].cpu(), points[:, 2].cpu(),
                    marker='.', c=col, s=s, alpha=alpha)
        ax3.set_xlim([-1, 1])
        ax3.set_ylim([-1, 1])

        ax4 = fig.add_subplot(224)
        ax4.scatter(points[:, 1].cpu(), points[:, 2].cpu(),
                    marker='.', c=col, s=s, alpha=alpha)
        ax4.set_xlim([-1, 1])
        ax4.set_ylim([-1, 1])

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def plot_gradients_per_layer(model):
    """Plots mean and max gradients per layer currently stored in model params. Inspired by https://github.com/alwynmathew/gradflow-check

    Args:
        model (torch model): Trained network
    """
    named_params = model.named_parameters()
    fig = plt.figure()
    avg_gradient, max_gradient, layers = [], [], []
    for name, parameter in named_params:
        if(parameter.requires_grad) and ("bias" not in name):
            layers.append(name)
            avg_gradient.append(parameter.grad.abs().mean())
            max_gradient.append(parameter.grad.abs().max())
    plt.bar(np.arange(len(max_gradient)),
            max_gradient, alpha=0.5, lw=1, color="lime")
    plt.bar(np.arange(len(max_gradient)),
            avg_gradient, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(avg_gradient)+1, lw=2, color="k")
    plt.xticks(range(0, len(avg_gradient), 1), layers, rotation="vertical")
    plt.xlim(left=-0.5, right=len(avg_gradient))
    # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layer Name")
    plt.ylabel("Average Gradient")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="lime", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    plt.tight_layout()
    plt.show()


def plot_model_vs_mascon_rejection(model, encoding, points, masses=None, N=100000, alpha=0.075, crop_p=1e-2, s=100, save_path=None, c=1., backcolor=[0.15, 0.15, 0.15]):
    """Plots both the mascon and model rejection in one figure for direct comparison

    Args:
        model (callable (N,M)->1): neural model for the asteroid. 
        encoding: the encoding for the neural inputs.
        points (2-D array-like): an (N, 3) array-like object containing the coordinates of the points
        masses (1-D array-like): a (N,) array-like object containing the values for the point masses
        N (int): number of points to be considered.
        views_2d (bool): activates also the 2d projections
        alpha (float): alpha for the visualization
        crop_p (float): all points below this density are rejected
        s (int): size of the non rejected points visualization
        save_path (str, optional): Pass to store plot, if none will display. Defaults to None.
        c (float, optional): Normalization constant. Defaults to 1.
        backcolor (list, optional): Plot background color. Defaults to [0.15, 0.15, 0.15].
    """

    # Mascon masses
    x = points[:, 0].cpu()
    y = points[:, 1].cpu()
    z = points[:, 2].cpu()

    s = 22000 / len(points)

    if masses is None:
        normalized_masses = s
    else:
        normalized_masses = masses / sum(masses)
        normalized_masses = (normalized_masses * s * len(x)).cpu()

    # Model samples
    points = torch.rand(N, 3) * 2 - 1
    nn_inputs = encoding(points)
    RHO = model(nn_inputs).detach() * c
    mask = RHO > (torch.rand(N, 1) + crop_p)
    RHO = RHO[mask]
    points = [[it[0].item(), it[1].item(), it[2].item()]
              for it, m in zip(points, mask) if m]
    if len(points) == 0:
        print("All points rejected! Plot is empty, try cropping less?")
        return
    points = torch.tensor(points)

    fig = plt.figure(dpi=150, facecolor=backcolor)
    ax = fig.add_subplot(221, projection='3d')
    ax.set_facecolor(backcolor)
    col = 'cornflowerblue'

    # And we plot it
    ax.scatter(x, y, z, color='k', s=normalized_masses, alpha=0.5)
    ax.scatter(points[:, 0].cpu(), points[:, 1].cpu(), points[:, 2].cpu(),
               marker='.', c=col, s=s, alpha=alpha)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=45., azim=125.)

    ax2 = fig.add_subplot(222)
    ax2.set_facecolor(backcolor)
    ax2.scatter(x, y, color='k', s=normalized_masses, alpha=0.5)
    ax2.scatter(points[:, 0].cpu(), points[:, 1].cpu(),
                marker='.', c=col, s=s, alpha=alpha)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])

    ax3 = fig.add_subplot(223)
    ax3.set_facecolor(backcolor)
    ax3.scatter(x, z, color='k', s=normalized_masses, alpha=0.5)
    ax3.scatter(points[:, 0].cpu(), points[:, 2].cpu(),
                marker='.', c=col, s=s, alpha=alpha)
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])

    ax4 = fig.add_subplot(224)
    ax4.set_facecolor(backcolor)
    ax4.scatter(y, z, color='k', s=normalized_masses, alpha=0.5)
    ax4.scatter(points[:, 1].cpu(), points[:, 2].cpu(),
                marker='.', c=col, s=s, alpha=alpha)
    ax4.set_xlim([-1, 1])
    ax4.set_ylim([-1, 1])

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
