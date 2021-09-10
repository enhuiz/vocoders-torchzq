import numpy as np
import matplotlib.pyplot as plt


def plot_to_numpy():
    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_tensor_to_numpy(tensor):
    plt.style.use("default")
    _, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(
        tensor.cpu().numpy(),
        aspect="auto",
        origin="lower",
        interpolation="none",
    )
    plt.colorbar(im, ax=ax)
    return plot_to_numpy()
