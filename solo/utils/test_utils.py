import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize

import torch
import seaborn as sns

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def vis_tsne(data, Y):
    num_classes = len(torch.unique(Y))
    color = sns.color_palette("hls", num_classes)
    Y = Y.numpy()
    data = data.numpy()
    data = TSNE(n_components=2, init='pca', random_state=0).fit_transform(data)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)

    for class_id in range(num_classes):
        data_class = data[Y == class_id]
        ax.scatter(data_class[:, 0], data_class[:, 1], color=color[class_id], marker='o',
                   label='class_' + str(class_id))
    ax.legend()
    ax.set_xlabel('Dim1')
    ax.set_ylabel('Dim2')

    plt.show()


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert (len(rater_a) == len(rater_b))
    if min_rating is None: min_rating = min(rater_a + rater_b)
    if max_rating is None: max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)] for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b): conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None: min_rating = min(ratings)
    if max_rating is None: max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings: hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa_(rater_a, rater_b, min_rating=None, max_rating=None):
    rater_a, rater_b = np.array(rater_a, dtype=int), np.array(rater_b, dtype=int)
    assert (len(rater_a) == len(rater_b))
    if min_rating is None: min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None: max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,min_rating, max_rating)
    num_ratings, num_scored_items = len(conf_mat), float(len(rater_a))
    hist_rater_a, hist_rater_b = histogram(rater_a, min_rating, max_rating), histogram(rater_b, min_rating, max_rating)
    numerator, denominator = 0.0, 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator

def test_core_vis(output_all, label, kwargs_vis):
    predict = output_all.argmax(dim=1)
    matrix = metrics.confusion_matrix(label.numpy(), predict.numpy())
    if 'Specific' in kwargs_vis:
        print('--Accuracy Specific--:  {}'.format(matrix.diagonal() / matrix.sum(axis=1)))
    if 'Average' in kwargs_vis:
        print('--Accuracy Average---: {}'.format(np.average(matrix.diagonal() / matrix.sum(axis=1))))

    if 'Total' in kwargs_vis:
        print('---Accuracy Total----: {}'.format(metrics.accuracy_score(label.numpy(), predict.numpy())))
    if 'Recall' in kwargs_vis:
        print('-------Recall--------: {}'.format(metrics.recall_score(label.numpy(), predict.numpy(), average='macro')))
    if 'Precision' in kwargs_vis:
        print('-----Precision-------: {}'.format(metrics.precision_score(label.numpy(), predict.numpy(), average='macro')))
    if 'F1_score' in kwargs_vis:
        print('------F1_score-------: {}'.format(metrics.f1_score(label.numpy(), predict.numpy(), average='macro')))
    if 'Kappa' in kwargs_vis:
        print('-------Kappa---------: {}'.format(metrics.cohen_kappa_score(label.numpy(), predict.numpy())))
    if 'QWK' in kwargs_vis:
        print('--------QWK----------: {}'.format(quadratic_weighted_kappa_(label.numpy(), predict.numpy())))

    if 'AUC' in kwargs_vis:
        output1 = (1 * output_all.float()).softmax(dim=1).numpy()
        label1 = label.numpy()
        y_test = label_binarize(label1, classes=[0, 1, 2, 3, 4])

        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(5):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], output1[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(5):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 5
        fpr_macro, tpr_macro = all_fpr, mean_tpr
        roc_auc_macro = metrics.auc(fpr_macro, tpr_macro)
        print('--------AUC----------: {}'.format(roc_auc_macro))

def valid_core(output_all, label):
    predict = output_all.argmax(dim=1)
    matrix = metrics.confusion_matrix(label.numpy(), predict.numpy())
    valid_acc = np.average(matrix.diagonal() / matrix.sum(axis=1))
    return valid_acc