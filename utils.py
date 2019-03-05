import numpy as np
from numpy.linalg import norm
import matplotlib.pylab  as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import math
import skimage
from skimage.io import imread


def read_gray(f):
    # im = skimage.img_as_float(imread(f))
    im = imread(f)
    return im


def list_files(dir_):
    files = []
    files.extend([f for f in sorted(os.listdir(dir_)) ])
    return files


def read_points(f):
    fh = open(f, "r")
    lines = fh.readlines()

    pts = []
    for line in lines:
        line = line.strip()
        pt = [float(x) for x in line.split(" ")]
        pts.append(pt)

    fh.close()
    return pts

def plot_2d_point_cloud(pts,
                        sks,
                        show=True,
                        show_axis=False,
                        in_u_sphere=False,
                        marker='.',
                        s=2,
                        alpha=.8,
                        figsize=(5, 5),
                        axis=None,
                        title=None,
                        filename=None,
                        colorize=None,
                        *args,
                        **kwargs):

    x = pts[:,0] 
    y = pts[:,1] 
    a = sks[:,0] 
    b = sks[:,1] 
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)        
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    if colorize is not None:
        cm = plt.get_cmap(colorize)
        col = [cm(float(i)/(x.shape[0])) for i in range(x.shape[0])]
        sc = ax.scatter(x, y, marker=marker, s=s, alpha=alpha, c=col, *args, **kwargs)
    else:
        sc = ax.scatter(x, y, marker=marker, s=s, alpha=alpha, *args, **kwargs)
        sc = ax.scatter(a, b, marker=marker, s=s, alpha=alpha, c="red", *args, **kwargs)

    # if not show_axis:
    #    plt.axis('off')

    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
    
    plt.close('all')
    return fig



def plot_3d_point_cloud(x,
                        y,
                        z,
                        show=True,
                        show_axis=False,
                        in_u_sphere=False,
                        marker='o',
                        s=10,
                        alpha=.8,
                        figsize=(5, 5),
                        elev=10,
                        azim=240,
                        axis=None,
                        title=None,
                        filename=None,
                        colorize=None,
                        *args,
                        **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')        
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    if colorize is not None:
        cm = plt.get_cmap(colorize)
        col = [cm(float(i)/(x.shape[0])) for i in range(x.shape[0])]
        sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, c=col, *args, **kwargs)
    else:
        sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)

    # sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        # Multiply with 0.7 to squeeze free-space.
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])  
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        #plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
    
    plt.close('all')
    return fig

