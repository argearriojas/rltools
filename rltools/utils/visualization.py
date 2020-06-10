import numpy as np
import matplotlib.pyplot as plt


def get_layout(desc):
    desc = np.asarray(desc, dtype='c')

    zeros = np.zeros_like(desc, dtype=np.float)
    walls = (desc == b'W').astype(np.float)
    holes = (desc == b'H').astype(np.float)
    candy = (desc == b'C').astype(np.float)
    nails = (desc == b'N').astype(np.float)

    out = np.array([zeros.T, zeros.T, zeros.T]).T

    # Walls: blue
    out += np.array([zeros.T, zeros.T, walls.T]).T
    # Holes: red
    out += np.array([holes.T, zeros.T, zeros.T]).T
    # Candy: orange
    out += np.array([candy.T, candy.T * 0.5, zeros.T]).T
    # Nails: silver
    out += np.array([nails.T, nails.T, nails.T]).T * 0.75

    return out


def display_map(desc):
    zeros = np.zeros_like(desc, dtype=np.float)
    goals = (desc == b'G').astype(np.float)
    start = (desc == b'S').astype(np.float)
    poten = (desc == b'S').astype(np.float)
    poten = np.array([float(l)/10 if l in b'0123456789' else 0.
                      for r in desc for l in r]).reshape(desc.shape)
    out = get_layout(desc)
    # Start: green
    out += np.array([zeros.T, start.T, zeros.T]).T
    # Goal: yellow
    out += np.array([goals.T, goals.T, zeros.T]).T
    # Potential: grayscale
    out += np.array([poten.T, poten.T, poten.T]).T

    plt.imshow(out)
    plt.show()


def plot_dist(desc, paths):
    zeros = np.zeros_like(desc, dtype=np.float)
    paths = paths / paths.max()
    paths = np.abs(paths.reshape(desc.shape))

    out = get_layout(desc)
    # Path: magenta
    out += np.array([paths.T, zeros.T, paths.T]).T

    out = np.minimum(out, 1.)
    plt.imshow(out)
    plt.show()


def plot_mdist(desc, M_list):
    holes = (desc==b'H').astype(float)
    n_axes = len(M_list)
    _, axes = plt.subplots(1, n_axes, sharey=True, figsize=(6*n_axes, 6*n_axes))
    for i, M in enumerate(M_list):
        M /= M.max()
        M = np.abs(M.reshape(desc.shape))
        axes[i].imshow(np.array([M.T, holes.T, M.T]).T)
    plt.show()
