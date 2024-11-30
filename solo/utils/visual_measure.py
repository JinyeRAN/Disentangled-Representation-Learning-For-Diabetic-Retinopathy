from sklearn.manifold import TSNE
import torch
import seaborn
import numpy as np
from matplotlib import pyplot as plt


class VisualizeBox():
    def __init__(self, ):
        self.tsne = TSNE(n_components=2, init='pca', random_state=0)
        self.legends = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']

    def filter(self, out:dict, num_samples:int, num_classes:int):
        points = [num_samples, ] * num_classes
        index = []
        for i, lbl in enumerate(out['target']):
            if points[lbl] > 0:
                points[lbl] = points[lbl] - 1
                index.append(i)

        output = {}
        for key in out.keys():
            temp = []
            for indx in index: temp.append(out[key][indx].unsqueeze(0))
            temp = torch.cat(temp, dim=0)
            output[key] = temp
        return output

    def vis_fn(
            self,
            data_col:dict,
            key: str,
            gap:list=None,
            name:str=None
    ):
        color = seaborn.color_palette("hls", len(gap))
        legends = [self.legends[i] for i in gap]
        Y, data = data_col['target'].numpy(), data_col[key].numpy()
        data = self.tsne.fit_transform(data)

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)

        for class_id, lbl, clr in zip(range(len(gap)), legends, color):
            data_class = data[Y == class_id]
            ax.scatter(data_class[:, 0], data_class[:, 1], color=clr, marker='o', label=lbl)

        ax.legend()
        ax.set_xlabel('Dim1')
        ax.set_ylabel('Dim2')
        plt.title(name)
        plt.show()
