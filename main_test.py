import json, torch, numpy, seaborn
import os
import torch.nn as nn

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

import numpy as np
from os import listdir
from os.path import join
from pathlib import Path
from sklearn import metrics
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from solo.methods import METHODS
from solo.utils.misc import make_contiguous
from solo.utils.auto_resumer import AutoResumer
from solo.data.dali_dataloader import ClassificationDALI

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

def valid_core_fn(output_all, label, method_type):
    predict = output_all.argmax(dim=1)
    matrix = metrics.confusion_matrix(label.numpy(), predict.numpy())
    QWK = quadratic_weighted_kappa_(label.numpy(), predict.numpy())
    valid_acc = numpy.average(matrix.diagonal() / matrix.sum(axis=1))
    special_valid_acc = matrix.diagonal() / matrix.sum(axis=1)
    print('{} average accuracy: {}; QWK: {}; special accuracy: {}'.format(method_type, valid_acc, QWK, special_valid_acc))
    print(matrix)

def vis_fn(data_train, data_test, Y_train, Y_test, method_type, checkpoint_file):
    ddata = torch.cat([data_train, data_test], dim=0)
    yy = torch.cat([Y_train, Y_test], dim=0)
    color = seaborn.color_palette("hls", 5)
    legends = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    data_train = TSNE(n_components=2, init='pca', random_state=0).fit_transform(data_train)
    data_test  = TSNE(n_components=2, init='pca', random_state=0).fit_transform(data_test)
    plt.figure(figsize=(27, 9))

    plt.subplot(1, 3, 1)
    for class_id, lbl, clr in zip(range(5), legends, color):
        data_class = data_train[Y_train == class_id]
        plt.scatter(data_class[:, 0], data_class[:, 1], color=clr, marker='o', label=lbl)
    plt.title("Train Data Visualize")
    plt.legend()

    plt.subplot(1, 3, 2)
    for class_id, lbl, clr in zip(range(5), legends, color):
        data_class = data_test[Y_test == class_id]
        plt.scatter(data_class[:, 0], data_class[:, 1], color=clr, marker='o', label=lbl)
    plt.title("Test Data Visualize")
    plt.legend()

    ddata = TSNE(n_components=2, init='pca', random_state=0).fit_transform(ddata)
    plt.subplot(1, 3, 3)
    for class_id, lbl, clr in zip(range(5), legends, color):
        data_class = ddata[yy == class_id]
        plt.scatter(data_class[:, 0], data_class[:, 1], color=clr, marker='o', label=lbl)
    plt.title("Test Data Visualize")
    plt.legend()

    # plt.suptitle('{}_{}'.format(checkpoint_file.split('/')[-1], method_type))
    plt.suptitle('{}_{}'.format(checkpoint_file, method_type))
    plt.tight_layout()
    # plt.savefig(checkpoint_file+method_type+'.png')
    plt.show()

def obtain_feats(model, loader):
    model.eval()
    feats, z, targets, logits, output = [], [], [], [], {}
    with torch.no_grad():
        for data, target, index in loader:
            with torch.cuda.amp.autocast(): kwargs: dict = model(data, target)

            targets.append(target.detach().cpu())
            if 'z' in kwargs.keys(): z.append(kwargs['z'].detach().cpu())
            if 'feats' in kwargs.keys(): feats.append(kwargs['feats'].detach().cpu())
            if 'logits' in kwargs.keys(): logits.append(kwargs['logits'].detach().cpu())

        output.update({'target': torch.cat(targets, dim=0)})
        if len(z) > 0: output.update({'z': torch.cat(z, dim=0)})
        if len(feats)>0: output.update({'feats': torch.cat(feats, dim=0)})
        if len(logits) > 0: output.update({'logits': torch.cat(logits, dim=0)})
    return output


def main(path_root, last:bool=True):
    dir_root = Path(path_root)
    with open(dir_root / 'args.json') as user_file: file_contents = user_file.read()
    cfg = OmegaConf.create(json.loads(file_contents))
    seed_everything(cfg.seed)

    model = METHODS[cfg.method](cfg)
    auto_resumer = AutoResumer(checkpoint_dir=join(cfg.checkpoint.dir, cfg.method), max_hours=100000)
    resume_from_checkpoint, wandb_id = auto_resumer.find_checkpoint(cfg)
    if wandb_id is None: dir_path = dir_root
    else: dir_path = Path(join(cfg.checkpoint.dir, cfg.method, wandb_id))

    if last: checkpoint_file = [dir_path / 'last.ckpt']
    else: checkpoint_file = [dir_path / f for f in listdir(dir_path) if f.endswith(".ckpt")][0:1]

    for ckpt_file in checkpoint_file:
        print('----------------------------------------------------------------------------')
        print('Task name: {}'.format(cfg.name))
        print('Process checkpoints: {}'.format(str(ckpt_file)))
        weight = torch.load(ckpt_file)['state_dict']
        model.load_state_dict(weight, strict=False)
        make_contiguous(model)
        model = model.to(memory_format=torch.channels_last).cuda()

        train_loader = ClassificationDALI(cfg, cfg.data.train_path).test_dataloader()
        test_loader = ClassificationDALI(cfg, cfg.data.val_path).test_dataloader()

        train_output: dict = obtain_feats(model, train_loader)
        test_output: dict  = obtain_feats(model, test_loader)

        method_types = ['z', 'feats']
        # # method_types = ['cls_mu','sps_mu', 'feats_fc', 'cls_mu','sps_mu','feats_rs']
        for mt in method_types:
            vis_fn(train_output[mt], test_output[mt], train_output['target'], test_output['target'], mt, str(ckpt_file))

        # valid_core_fn(train_output['logits'], train_output['target'], 'Train')
        # valid_core_fn(test_output['logits'], test_output['target'], 'Test')



if __name__ == "__main__":
    main('./trained_models/mocov2/E800/5tugb1vb', last=False) # self-supervised
    # main('./trained_models/mocodisentangle/mj5sp4kj', last=False) # self-supervised_de
    # main('./trained_models/supervised/supervised-epoch400', last=False) # supervised