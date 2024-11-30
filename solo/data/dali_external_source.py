import torch
import random
import numpy as np


class ExternalInputGpuIterator(object):
    def __init__(self, samples, targets, num_classes, batch_size, balance):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.samples = samples
        self.targets = targets
        self.balance = balance
        self.data_len = len(self.samples)

        class_weights = self.cal_class_weights()
        w0 = torch.as_tensor(class_weights, dtype=torch.double)
        self.sample_weight = torch.zeros(self.data_len, dtype=torch.double)
        for i, _class in enumerate(self.targets): self.sample_weight[i] = w0[_class]

    def cal_class_weights(self):
        classes_idx = list(range(self.num_classes))
        class_count = [self.targets.count(i) for i in classes_idx]
        weights = [self.data_len / class_count[i] for i in classes_idx]
        min_weight = min(weights)
        class_weights = [weights[i] / min_weight for i in classes_idx]
        return class_weights

    def __iter__(self):
        self.i = 0
        self.n = len(self.samples)
        if self.balance:
            indx = torch.multinomial(self.sample_weight, self.data_len, replacement=True).tolist()
        else:
            indx = list(torch.arange(0, len(self.samples), 1).numpy())
        if self.balance: random.shuffle(indx) # reide
        else: random.shuffle(indx)
        self.index = indx
        self.labels = [self.targets[i] for i in indx]
        self.points = [self.samples[i] for i in indx]
        return self

    def __len__(self):
        return len(self.samples)

    # def __next__(self):
    #     batch, labels = [], []
    #     if self.i >= self.data_len:
    #         self.__iter__()
    #         raise StopIteration
    #
    #     leave_num = self.n - self.i
    #     current_batch_size = min(self.batch_size, leave_num)
    #     for _ in range(current_batch_size):
    #         filedir = self.points[self.i]
    #         lbl = self.labels[self.i]
    #         f = open(filedir, 'rb')
    #
    #         batch.append(np.frombuffer(f.read(), dtype = np.uint8))
    #         labels.append(np.array([lbl], dtype = np.uint8))
    #         self.i = self.i + 1
    #     return batch, labels

    def __next__(self):
        batch, labels, filename = [], [], []
        if self.i >= self.data_len:
            self.__iter__()
            raise StopIteration

        leave_num = self.n - self.i
        current_batch_size = min(self.batch_size, leave_num)
        for _ in range(current_batch_size):
            index = self.index[self.i]
            filedir = self.points[self.i]
            lbl = self.labels[self.i]
            f = open(filedir, 'rb')

            filename.append(np.array([index], dtype = np.uint8))
            batch.append(np.frombuffer(f.read(), dtype = np.uint8))
            labels.append(np.array([lbl], dtype = np.uint8))
            self.i = self.i + 1
        return batch, labels, filename