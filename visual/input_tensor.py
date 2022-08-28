import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def show_images(image_batch,bs):
    """

    :param image_batch: ndarray BHWC
    :param bs:
    :return:
    """
    columns = 4
    rows = (bs + 1) // (columns)
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch[j])
    plt.show()