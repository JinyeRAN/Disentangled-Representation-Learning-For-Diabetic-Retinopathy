import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn


def plot_ind(
        data: torch.tensor,
        Y: torch.tensor,
        gap: list,
        data_en: torch.tensor = None,
        Y_en: torch.tensor = None,
        gap_en: list = None,
):
    color = seaborn.color_palette("hls", 5)
    legend = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    data, Y, ptr = data.numpy(), Y.numpy(), data.shape[0]
    color, legends = [color[i] for i in gap], [legend[i] for i in gap]
    if not data_en is None and not Y_en is None and not gap_en is None:
        data_en, Y_en = data_en.numpy(), Y_en.numpy()
        data = np.vstack((data, data_en))

    tsne_data = TSNE(n_components=2, init='pca', random_state=0).fit_transform(data)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    data = tsne_data[0:ptr, :]

    for class_id, lbl, clr in zip(gap, legends, color):
        data_class = data[Y == class_id]
        ax.scatter(data_class[:, 0], data_class[:, 1], color=clr, marker='o', label=lbl)

    if not data_en is None and not Y_en is None and not gap_en is None:
        data_en = tsne_data[ptr:, :]
        color_en, legends_en = [color[i] for i in gap_en], [legend[i] + '_enhance' for i in gap_en]
        for class_id, lbl, clr in zip(gap_en, legends_en, color_en):
            data_class = data_en[Y_en == class_id]
            ax.scatter(data_class[:, 0], data_class[:, 1], color=clr, marker='+', label=lbl)

    # plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.legend(prop={'size':11,})# 'weight': 'bold'})
    plt.show()

def plot_ind_x(
        data: torch.tensor,
        Y: torch.tensor,
        gap: list,
        data_en: torch.tensor = None,
        Y_en: torch.tensor = None,
        gap_en: list = None,
):
    color = seaborn.color_palette("hls", 5)
    legend = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    data, Y, ptr = data.numpy(), Y.numpy(), data.shape[0]
    color, legends = [color[i] for i in gap], [legend[i] for i in gap]
    if not data_en is None and not Y_en is None and not gap_en is None:
        data_en, Y_en = data_en.numpy(), Y_en.numpy()
        data = np.vstack((data, data_en))

    tsne_data = TSNE(n_components=2, init='pca', random_state=0).fit_transform(data)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    data = tsne_data[0:ptr, :]

    for class_id, lbl, clr in zip(gap, legends, color):
        data_class = data[Y == class_id]
        ax.scatter(data_class[:, 0], data_class[:, 1], color=clr, marker='o', label=lbl)

    if not data_en is None and not Y_en is None and not gap_en is None:
        data_en = tsne_data[ptr:, :]
        color_en, legends_en = [color[i] for i in gap_en], [legend[i] + '_generate' for i in gap_en]
        for class_id, lbl, clr in zip(gap_en, legends_en, color_en):
            data_class = data_en[Y_en == class_id]
            ax.scatter(data_class[:, 0], data_class[:, 1], color=clr, marker='+', label=lbl)

    # plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.legend(prop={'size':11,})# 'weight': 'bold'})
    plt.show()



if __name__ == '__main__':
    data = torch.load('./total_data_1.dat')
    # plot_ind(data['cls_mu'], data['targets_rs'], gap=[0, 1, 2, 3, 4])
    # plot_ind(data['sps_mu'], data['targets_rs'], gap=[0, 1, 2, 3, 4])
    plot_ind(data['feats'], data['targets'], gap=[0, 1, 2, 3, 4])
    plot_ind(data['feats_rs'], data['targets_rs'], gap=[0, 1, 2, 3, 4])

    # data = torch.load('./work_set_part.dat')
    # plot_ind_x(data['cls_mu'], data['targets_rs'], gap=[0, 1, 2, 3, 4])
    # plot_ind_x(data['sps_mu'], data['targets_rs'], gap=[0, 1, 2, 3, 4])
    # plot_ind_x(
    #     data=data['feats_rs'], Y=data['targets_rs'], gap=[0, 1, 2, 3, 4],
    #     data_en=data['sweap_swp'], Y_en=data['targets_swp'], gap_en=[1, 3]
    # )
    # plot_ind_x(
    #     data=data['feats_rs'], Y=data['targets_rs'], gap=[0, 1, 2, 3, 4],
    #     data_en=data['feats_mpl'], Y_en=data['targets_mpl'], gap_en=[3]
    # )
