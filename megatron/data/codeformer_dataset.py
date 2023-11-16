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
        task,
        name,
        indexed_dataset,
        indexed_labels,
        data_prefix,
        num_epochs,
        max_num_samples,
        masked_lm_prob,
        max_seq_length,
        max_seq_length_dec,
        max_chunk_num,
        max_label_length,
        seed
    ):

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.max_seq_length_dec = max_seq_length_dec
        self.max_chunk_num = max_chunk_num
        self.max_label_length = max_label_length

        # Dataset.
        self.indexed_dataset = indexed_dataset
        self.indexed_labels = indexed_labels
        # pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)

        # Build the samples mapping.
        # TODO may be use helpers.build_mapping (see language_model.py) for building sample mapping
        args = get_args()
        # TODO you should to change this, to exclude indexed_labels.
        num_samples = indexed_dataset.doc_idx.shape[0]-1
        self.samples_mapping = create_sample_mapping_cf(num_samples, seed, max_num_samples, name, data_prefix, args.separate_split_files)
        # self.n = 1

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.task = task
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
        self.first = 0
        self.args = args
        assert len(self.sentinel_tokens) > 0, "Provide the argument --vocab-extra-ids 100 to the script"

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        # pydevd_pycharm.settrace("localhost", ports=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)
        doc_index = self.samples_mapping[idx]
        sent_idx_start = self.indexed_dataset.doc_idx[doc_index]
        sent_idx_end = self.indexed_dataset.doc_idx[doc_index + 1]
        if self.task == 'method_naming':
            label = self.indexed_labels[doc_index]
        else:
            label = None
        sample = []
        for index in range(sent_idx_start, sent_idx_end):
            # I do not do any truncation or padding here
            # everything is already done in data preprocessing
            sentence = self.indexed_dataset[index]
            sample.append(sentence)

        # Warmup - added first maximal size batch to be sure that all other batche will fit into the model
        # TODO turn on warmup
        # if self.first < 2*self.args.micro_batch_size:
        #     sample = sample + (self.max_chunk_num-len(sample))*[(self.args.max_sent_length+2)*[1]]
        #     self.first += 1

        return build_training_sample(
            self.task,
            sample,
            label,
            self.max_chunk_num,
            self.cls_id,
            self.sep_id,
            self.mask_id,
            pad_id=self.pad_id,
        )

def build_training_sample(
    task,
    sample,
    label,
    max_chunk_num,
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
        max_chunk_num: Maximum number of subtrees in the method.
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
    sample = sample[:max_chunk_num]
    num_chunks = len(sample)
    flattened_sample = np.concatenate(sample, axis=0, dtype=np.int64)
    sample = np.array(sample, dtype=np.int64)
    enc_mask = [make_attention_mask(chunk, chunk, pad_id) for chunk in sample]
    if task == "method_naming":
        label = np.array(label, dtype=np.int64)
        # TODO I removed label mask (label != pad_id)
        loss_mask = (label != -100).astype(np.int64)
        dec_mask = make_attention_mask(label, label, pad_id)
        dec_mask = dec_mask * make_history_mask(label.shape[0])
    elif task == "language_modeling":
        # TODO make properly for language modeling
        label = np.array(7*[0], dtype=np.int64)
        # loss_mask = np.array(7*[0], dtype=np.int64)
        loss_mask = (sample != pad_id).astype(np.int64)
        dec_mask = [np.ones((sample[0].shape[0]+1, sample[0].shape[0]+1), dtype=np.int64) for mask in enc_mask]
        for mask in dec_mask:
            mask[1:,1:] = mask[1:,1:] * make_history_mask(sample[0].shape[0])
            mask[:1, 1:] = 0


    ## Mask for the second encoder and enc_dec_mask are built in collate_fn in data_sampler
    train_sample = {
        "docs_enc": flattened_sample,
        "sent_nums": num_chunks,
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


def make_history_mask(length):
    # length = block.shape[0]
    arange = np.arange(length)
    history_mask = (
        arange[
            None,
        ]
        <= arange[:, None]
    )
    history_mask = history_mask.astype(np.int64)
    return history_mask

def create_sample_mapping_cf(num_samples, seed, max_num_samples, name, data_prefix, separate_split_files=False):
    epochs = max_num_samples//num_samples
    resid_n = max_num_samples - epochs*num_samples
    if epochs == 0:
        epochs = 1
        resid_n = 0
    np.random.seed(seed)
    samples_mapping = []
    sample_range = np.arange(num_samples)
    for _ in range(epochs):
        shuffled_copy = np.random.permutation(sample_range)
        samples_mapping.append(shuffled_copy)
    resid = np.random.choice(sample_range, resid_n, replace=False)
    samples_mapping = samples_mapping + [resid]
    samples_mapping = np.concatenate(samples_mapping)

    if not separate_split_files:
        # TODO it would work inproperly if we input single file, not splitted into train-val-test
        split_set_sent_indices_file = f"{name}_set_indices.npy"
        split_set_sent_indices_file = os.path.join(os.path.dirname(data_prefix), split_set_sent_indices_file)
        np.save(split_set_sent_indices_file, sample_range)

    split_sent_indices_file = f"{name}_indices.npy"
    split_sent_indices_file = os.path.join(os.path.dirname(data_prefix), split_sent_indices_file)
    np.save(split_sent_indices_file, samples_mapping)

    return samples_mapping