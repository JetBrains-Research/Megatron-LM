import os
import sys
import torch
from functools import partial

# from commode_utils.metrics import SequentialF1Score
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from .chrf import ChrF
# from codeformer_utils.vendor.codeformer import ChrF
# from torchmetrics import MetricCollection

def recall(prediction, target, ignored_indices=[]):
    prediction_flat, target_flat = prediction.cpu().view(-1), target.cpu().view(-1)
    mask = torch.ones_like(target_flat, dtype=torch.bool)
    for idx in ignored_indices:
        mask = mask & (target_flat != idx)
    prediction_flat, target_flat = prediction_flat[mask], target_flat[mask]
    recall = torch.sum(prediction_flat == target_flat)/len(prediction_flat)
    return recall


class CFMetrics():
    def __init__(self, tokenizer, log_file=None):

        pad_id = tokenizer.pad
        bos_id = tokenizer.bos_token_id
        # eos_id = tokenizer.eos_token_id
        unk_id = tokenizer.tokenizer.unk_token_id
        self.tokenizer = tokenizer.tokenizer

        self.recall = partial(recall, ignored_indices=[pad_id, bos_id, unk_id])
        self.chrf = ChrF(tokenizer.tokenizer)
        self.chrf.log_file=log_file

    def calc_metrics(self, logits, label_tokens, prefix=""):
        prediction = logits.argmax(-1)
        recall = self.recall(prediction=prediction, target=label_tokens)
        result = {f"{prefix}recall": recall}
        result["chrf"] = self.chrf(prediction, label_tokens)

        return result



