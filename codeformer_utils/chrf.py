import torch
from sacrebleu import CHRF
from torchmetrics import Metric

class ChrF(Metric):
    def __iter__(self):
        pass

    def __init__(self, vocab, **kwargs):
        super().__init__(**kwargs)
        self.__vocab = vocab
        self.__chrf = CHRF()
        self.log_file = None

        # Metric states
        self.add_state(
            "chrf", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        """Calculated ChrF metric on predicted tensor w.r.t. target tensor.

        :param predicted: [pred seq len; batch size] -- tensor with predicted tokens
        :param target: [target seq len; batch size] -- tensor with ground truth tokens
        :return:
        """
        if predicted.shape[1] != target.shape[1]:
            raise ValueError(
                f"Wrong batch size for prediction (expected: {target.shape[1]}, actual: {predicted.shape[1]})"
            )

        # TODO correct to be able to use it in method-naming
        predicted_str = self.__vocab.batch_decode(predicted.t(), skip_special_tokens=True)
        target_str = self.__vocab.batch_decode(target.t(), skip_special_tokens=True)
        if self.log_file is not None:
            f = open(self.log_file, 'a', encoding='utf-8')
        for pred, targ in zip(predicted_str, target_str):
            if targ == "":
                # Empty target string mean that the original string encoded only with <UNK> token
                continue
            self.chrf += self.__chrf.sentence_score(pred, [targ]).score
            self.count += 1

            if self.log_file is not None:
                f.write(f'{targ} ---- {pred} \n')

        f.close()


    def compute(self) -> torch.Tensor:
        return self.chrf / self.count
