# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Codeformer Style dataset. Build on the basis of T5 dataset"""

import numpy as np
import torch
import os

from megatron import get_tokenizer
from megatron import get_args

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
        max_sent_num,
        max_label_length,
        seed,
    ):

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.max_seq_length_dec = max_seq_length_dec
        self.max_sent_num = max_sent_num
        self.max_label_length = max_label_length

        # Dataset.
        self.indexed_dataset = indexed_dataset
        self.indexed_labels = indexed_labels
        # pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)

        # Build the samples mapping.
        # TODO may be use helpers.build_mapping (see language_model.py) for building sample mapping
        args = get_args()
        self.samples_mapping = create_sample_mapping_cf(indexed_labels, seed, max_num_samples, name, data_prefix, args.separate_split_files)
        # self.n = 1

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.cls_id = tokenizer.cls
        self.sep_id = tokenizer.sep
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad
        args.pad_id = self.pad_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.sentinel_tokens = tokenizer.additional_special_tokens_ids
        assert len(self.sentinel_tokens) > 0, "Provide the argument --vocab-extra-ids 100 to the script"

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        # pydevd_pycharm.settrace("localhost", port=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)
        # if self.name == 'train':
        #     print(self.n, idx)
        #     self.n += 1
        doc_index = self.samples_mapping[idx]
        sent_idx_start = self.indexed_dataset.doc_idx[doc_index]
        sent_idx_end = self.indexed_dataset.doc_idx[doc_index + 1]
        label = self.indexed_labels[doc_index]
        sample = []
        for index in range(sent_idx_start, sent_idx_end):
            # I do not do any truncation or padding here
            # everything is already done in data preprocessing
            sentence = self.indexed_dataset[index]
            sample.append(sentence)

        return build_training_sample(
            sample,
            label,
            self.max_sent_num,
            self.cls_id,
            self.sep_id,
            self.mask_id,
            pad_id=self.pad_id,
        )


def build_training_sample(
    sample,
    label,
    max_sent_num,
    cls_id,
    sep_id,
    mask_id,
    pad_id,
    bos_id=None,
    eos_id=None,
):
    """Build training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        label: Method labels - a list token ids.
        max_sent_num: Maximum number of subtrees in the method.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        bos_id: start of decoder example id
        eos_id: end of generation id
    """
    # pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)
    pad_id = 0
    sample = sample[:max_sent_num]
    num_sent = len(sample)
    # Padding.
    # sample = sample + (max_sent_num - num_sent) * [0 * sample[0]]
    flattened_sample = np.concatenate(sample, axis=0, dtype=np.int64)
    # label is already padded
    label = np.array(label, dtype=np.int64)
    loss_mask = (label != pad_id).astype(np.int64)

    # Create attention masks and padding them
    enc_mask = [make_attention_mask(sentence, sentence, pad_id) for sentence in sample]
    dec_mask = make_attention_mask(label, label, pad_id)
    dec_mask = dec_mask * make_history_mask(label)
    ## Mask for the second encoder is build in collate_fn in data_sampler

    train_sample = {
        "docs_enc": flattened_sample,
        "sent_nums": num_sent,
        "labels": label,
        "loss_mask": loss_mask,
        "enc_mask": np.stack(enc_mask).astype(np.int64),
        "dec_mask": np.stack(dec_mask).astype(np.int64),
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

def create_sample_mapping_cf(indexed_labels, seed, max_num_samples, name, data_prefix, separate_split_files=False):
    epochs = max_num_samples//(indexed_labels.doc_idx.shape[0]-1)
    resid_n = max_num_samples - epochs*(indexed_labels.doc_idx.shape[0]-1)
    if epochs == 0:
        epochs = 1
        resid_n = 0
    np.random.seed(seed)
    samples_mapping = []
    for _ in range(epochs):
        shuffled_copy = np.random.permutation(indexed_labels.doc_idx[:-1])
        samples_mapping.append(shuffled_copy)
    resid = np.random.choice(indexed_labels.doc_idx[:-1], resid_n, replace=False)
    samples_mapping = samples_mapping + [resid]
    samples_mapping = np.concatenate(samples_mapping)

    if not separate_split_files:
        split_set_sent_indices_file = f"{name}_set_indices.npy"
        split_set_sent_indices_file = os.path.join(os.path.dirname(data_prefix), split_set_sent_indices_file)
        np.save(split_set_sent_indices_file, indexed_labels.doc_idx[:-1])

    split_sent_indices_file = f"{name}_indices.npy"
    split_sent_indices_file = os.path.join(os.path.dirname(data_prefix), split_sent_indices_file)
    np.save(split_sent_indices_file, samples_mapping)

    return samples_mapping