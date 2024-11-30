import torch
import torch.nn.functional as F


class WeightedKNNClassifier(object):
    def __init__(
        self,
        k: int = 5,
        T: float = 0.07,
        max_distance_matrix_size: int = int(5e6),
        distance_fx: str = "cosine",
        epsilon: float = 0.00001,
    ):

        super().__init__()

        self.k = k
        self.T = T
        self.max_distance_matrix_size = max_distance_matrix_size
        self.distance_fx = distance_fx
        self.epsilon = epsilon

        self.train_features = []
        self.train_targets = []
        self.test_features = []
        self.test_targets = []

    def update(
        self,
        train_features: torch.Tensor = None,
        train_targets: torch.Tensor = None,
        test_features: torch.Tensor = None,
        test_targets: torch.Tensor = None,
    ):
        assert (train_features is None) == (train_targets is None)
        assert (test_features is None) == (test_targets is None)

        if train_features is not None and train_targets is not None:
            assert train_features.size(0) == train_targets.size(0)
            self.train_features.append(train_features.detach())
            self.train_targets.append(train_targets.detach())

        if test_features is not None and test_targets is not None:
            assert test_features.size(0) == test_targets.size(0)
            self.test_features.append(test_features.detach())
            self.test_targets.append(test_targets.detach())

    @torch.no_grad()
    def compute(self):
        train_features = torch.cat(self.train_features, dim=0)
        train_targets = torch.cat(self.train_targets, dim=0)
        test_features = torch.cat(self.test_features, dim=0)
        test_targets = torch.cat(self.test_targets, dim=0)

        if self.distance_fx == "cosine":
            train_features = F.normalize(train_features)
            test_features = F.normalize(test_features)

        num_classes = torch.unique(test_targets).numel()
        num_train_images = train_targets.size(0)
        num_test_images = test_targets.size(0)

        k = min(self.k, num_train_images)
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)

        # get the features for test images
        features = test_features[0 : num_test_images, :]
        targets = test_targets[0 : num_test_images]
        batch_size = targets.size(0)

        # calculate the dot product and compute top-k neighbors
        if self.distance_fx == "cosine":
            similarities = torch.mm(features, train_features.t())
        elif self.distance_fx == "euclidean":
            similarities = 1 / (torch.cdist(features, train_features) + self.epsilon)
        else:
            raise NotImplementedError

        similarities, indices = similarities.topk(k, largest=True, sorted=True)
        candidates = train_targets.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

        if self.distance_fx == "cosine":
            similarities = similarities.clone().div_(self.T).exp_()

        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                similarities.view(batch_size, -1, 1),
            ),
            1,
        )
        probs = F.normalize(probs, dim=1)

        self.reset()
        return probs, targets

    def reset(self):
        self.train_features.clear()
        self.train_targets.clear()
        self.test_features.clear()
        self.test_targets.clear()