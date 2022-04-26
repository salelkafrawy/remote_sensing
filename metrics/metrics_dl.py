import os
import sys
import inspect
from pathlib import Path

CURR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURR_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, CURR_DIR)

import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import Metric


from metrics_torch import top_k_error_rate_top_30_set


class TopK_error(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("top_k", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.FloatTensor([0]), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor, preds: torch.Tensor):

        probas = torch.nn.functional.softmax(preds, dim=0)
        self.top_k += top_k_error_rate_top_30_set(probas, target)
        self.total += 1  # target.shape[0]

    def compute(self):
        return self.top_k / self.total


class AdaptiveTopK_Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.FloatTensor([0]), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor, preds: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        non_zero_counts = torch.count_nonzero(target, dim=1)
        for i, elem in enumerate(target):
            ki = non_zero_counts[i]
            v_pred, i_pred = torch.topk(preds[i], k=ki)
            v_targ, i_targ = torch.topk(elem, k=ki)
            if ki == 0:
                self.correct += 1
            else:
                self.correct += (
                    len(
                        set(i_pred.cpu().numpy()).intersection(
                            set(i_targ.cpu().numpy())
                        )
                    )
                    / ki
                )
        self.total += target.shape[0]

    def compute(self):
        return (self.correct / self.total).float()


def get_metric(metric):
    """Returns the transform function associated to a
    transform_item listed in opts.data.transforms ; transform_item is
    an addict.Dict
    """

    if metric.name == "topk-error" and not metric.ignore is True:
        return TopK_error()

    elif metric.name == "topk-accuracy" and not metric.ignore is True:
        return AdaptiveTopK_Accuracy()

    elif metric.ignore is True:
        return None

    raise ValueError("Unknown metric_item {}".format(metric))


def get_metrics(opts):
    metrics = []

    for m in opts.losses.metrics:
        metrics.append((m.name, get_metric(m), m.scale))
    metrics = [(a, b, c) for (a, b, c) in metrics if b is not None]
    print(f"metrics: {metrics}")
    return metrics
