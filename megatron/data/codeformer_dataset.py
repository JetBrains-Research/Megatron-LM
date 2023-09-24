# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Codeformer Style dataset. Build on the basis of T5 dataset"""

import collections

import numpy as np
import torch

from megatron import get_tokenizer
from megatron.data.dataset_utils import create_masked_lm_predictions, get_samples_mapping

import pydevd_pycharm

PORT_DEBUG = 2000


class CodeformerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,
        indexed_dataset,
        indexed_labels,
        data_prefix,
        num_epochs,
        max_num_samples,
        masked_lm_prob,
        max_seq_length,
        max_seq_length_dec,
        max_doc_length,
        max_sent_num,
        max_sent_length,
        max_label_length,
        short_seq_prob,
        seed,
    ):

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.max_seq_length_dec = max_seq_length_dec
        self.max_doc_length = max_doc_length
        self.max_sent_num = max_sent_num
        self.max_sent_length = max_sent_length
        self.max_label_length = max_label_length

        # Dataset.
        self.indexed_dataset = indexed_dataset
        self.indexed_labels = indexed_labels
        # pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)

        # Build the samples mapping.
        ## TODO Here we need to realize train-val-test split!
        self.samples_mapping = get_samples_mapping(
            self.indexed_dataset,
            data_prefix,
            num_epochs,
            max_num_samples,
            self.max_seq_length - 2,  # account for added tokens
            short_seq_prob,
            self.seed,
            self.name,
            False,
        )

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.cls_id = tokenizer.cls
        self.sep_id = tokenizer.sep
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.sentinel_tokens = tokenizer.additional_special_tokens_ids
        assert len(self.sentinel_tokens) > 0, "Provide the argument --vocab-extra-ids 100 to the script"

    def __len__(self):
        return len(self.indexed_dataset.doc_idx) - 1

    def __getitem__(self, idx):
        # if idx == 655:
        #     pydevd_pycharm.settrace("localhost", port=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)
        ## TODO 2. Check about EOS/BOS tokens
        # with open("some_output.txt", mode="a") as f:
        #     f.write(str(idx))
        #     f.write("\n")
        doc_idx_start = self.indexed_dataset.doc_idx[idx]
        doc_idx_end = self.indexed_dataset.doc_idx[idx + 1]
        sample = []
        for index in range(doc_idx_start, doc_idx_end):
            sentence = self.indexed_dataset[index][: self.max_sent_length]
            sentence = np.pad(sentence, (self.pad_id, self.max_sent_length - len(sentence)))
            sample.append(sentence)
        label = self.indexed_labels[idx]
        # self.max_doc_length = (self.max_doc_length // self.max_sent_length) * self.max_sent_length
        return build_training_sample(
            sample,
            label,
            self.max_label_length,
            self.max_sent_num,
            self.cls_id,
            self.sep_id,
            self.mask_id,
            pad_id=self.pad_id,
        )


def build_training_sample(
    sample,
    label,
    max_label_length,
    max_sent_num,
    cls_id,
    sep_id,
    mask_id,
    pad_id,
    masked_lm_prob=None,
    bos_id=None,
    eos_id=None,
    sentinel_tokens=None,
):
    """Build training sample.
    ## TODO rewrite doc

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        bos_id: start of decoder example id
        eos_id: end of generation id
    """
    pad_id = 0
    sample = sample[:max_sent_num]
    num_sent = len(sample)
    # Padding.
    sample = sample + (max_sent_num - num_sent) * [0 * sample[0]]
    flattened_sample = np.concatenate(sample, axis=0, dtype=np.int64)
    label = label[:max_label_length]
    label = np.pad(label, (pad_id, max_label_length - len(label))).astype(np.int64)
    loss_mask = (label != pad_id).astype(np.int64)

    # Create attention masks and padding them
    enc_mask = [make_attention_mask(sentence, sentence, pad_id) for sentence in sample]
    dec_mask = make_attention_mask(label, label, pad_id)
    dec_mask = dec_mask * make_history_mask(label)
    ## Mask for the second encoder
    sent_flags = np.array(num_sent * [1] + (max_sent_num - num_sent) * [pad_id], dtype=np.int64)
    enc_dec_mask = make_attention_mask(label, sent_flags, pad_id)
    sent_mask = make_attention_mask(sent_flags, sent_flags, pad_id)

    train_sample = {
        "docs_enc": flattened_sample,
        "sent_nums": num_sent,
        "labels": label,
        "loss_mask": loss_mask,
        "enc_mask": np.stack(enc_mask).astype(np.int64),
        "dec_mask": np.stack(dec_mask).astype(np.int64),
        "enc_dec_mask": enc_dec_mask,
        "sent_mask": sent_mask,
    }

    return train_sample


def make_attention_mask(source_block, target_block, pad_id):
    """
    Returns a 2-dimensional (2-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[None, :] != pad_id) * (source_block[:, None] != pad_id)
    mask = mask.astype(np.int64)
    # (source_length, target_length)
    return mask


def make_history_mask(block):
    length = block.shape[0]
    arange = np.arange(length)
    history_mask = (
        arange[
            None,
        ]
        <= arange[:, None]
    )
    history_mask = history_mask.astype(np.int64)
    return history_mask
