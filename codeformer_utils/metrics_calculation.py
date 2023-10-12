import os
import sys
from commode_utils.metrics import SequentialF1Score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from codeformer_utils.vendor.codeformer import ChrF
from torchmetrics import MetricCollection

class CFMetrics():
    def __init__(self, tokenizer):

        pad_id = tokenizer.pad
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        unk_id = tokenizer.tokenizer.unk_token_id
        self.tokenizer = tokenizer.tokenizer

        metrics = {
            "f1": SequentialF1Score(
                pad_idx=pad_id,
                eos_idx=eos_id,
                ignore_idx=[bos_id, unk_id],
            )
        }
        metrics.update({"chrf": ChrF(tokenizer.tokenizer)})
        self.metrics = MetricCollection(metrics)

    def calc_metrics(self, logits, label_tokens, prefix=""):
        prediction = logits.argmax(-1)
        pred_txt = self.tokenizer.batch_decode(prediction.t(), skip_special_tokens=True)
        label_txt = self.tokenizer.batch_decode(label_tokens.t(), skip_special_tokens=True)
        with open('out.txt', 'a') as f:
            for lab, pred in zip(label_txt, pred_txt):
                f.write(f'{lab} ---- {pred} \n')
        metric = self.metrics["f1"](prediction, label_tokens)
        result = {f"{prefix}f1": metric.f1_score,
                f"{prefix}precision": metric.precision,
                f"{prefix}recall": metric.recall}
        result["chrf"] = self.metrics["chrf"](prediction, label_tokens)

        return result



