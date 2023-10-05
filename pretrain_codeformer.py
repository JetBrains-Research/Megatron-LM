# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain T5"""

from functools import partial

import torch

from megatron import get_args, get_timers, print_rank_0
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import CodeformerModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args
import pydevd_pycharm

# TODO FINAL delete all pydevd_pycharm instances at the end
# import pydevd_pycharm
# pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)

# TODO FINAL Change description
"""
Pipeline parallelism for T5
===========================

T5 is a model architecture with both encoder and decoder blocks.
Consequently, pipeline parallelism is implemented slightly differently
compared to architectures like GPT and BERT.

In particular, when pipeline_model_parallel_world_size > 1, each stage
either executes an encoder block or a decoder block. The
--pipeline-model-parallel-split-rank argument controls the rank at which
the split happens: all ranks lower than this argument execute the
encoder block, and all ranks equal to or higher than this argument value
execute the decoder block.

In the encoder section of the model, only one tensor is sent downstream:
the intermediate encoder_hidden_state. In the decoder section of the
model, two tensors are sent downstream in the forward pass: the fully
computed encoder_hidden_state, and the intermediate decoder_hidden_state.

In particular, these are the shapes of the tensors sent between
different workers:
    If rank is in decoder section:
        intermediate decoder_hidden_state (pre-transpose),
        complete encoder_hidden_state (post-transpose).
    If rank is at boundary between encoder and decoder sections:
        complete encoder_hidden_state (post-transpose).
    If rank is in encoder section:
        intermediate encoder_hidden_state (pre-transpose).

Additionally, we have code in the backward_step function in schedules.py
to accumulate the encoder_hidden_state gradient across skip connections
(encoder_hidden_state fed in as input to each layer in the decoder).
"""

PORT_DEBUG = 2000
try:
    pass
    # pydevd_pycharm.settrace("localhost", port=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)
except Exception as e:
    print("No debugging")


def model_provider(pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
    """Build the model."""
    # args = get_args()
    # pydevd_pycharm.settrace("localhost", port=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)
    print_rank_0("building Codeformer model ...")
    config = core_transformer_config_from_args(get_args())
    model = CodeformerModel(
        config=config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
    )
    return model


def get_batch(data_iterator):
    """Build the batch."""
    keys = ["docs_enc", "sent_nums", "labels", "loss_mask", "enc_mask", "dec_mask", "enc_dec_mask", "sent_mask"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    docs_enc = data_b["docs_enc"].long()
    labels = data_b["labels"].long()
    sent_nums = data_b["sent_nums"].long()
    loss_mask = data_b["loss_mask"].float()

    enc_mask = data_b["enc_mask"] < 0.5
    sent_mask = data_b["sent_mask"] < 0.5
    enc_dec_mask = data_b["enc_dec_mask"] < 0.5
    dec_mask = data_b["dec_mask"] < 0.5

    return docs_enc, sent_nums, labels, loss_mask, enc_mask, sent_mask, enc_dec_mask, dec_mask

def loss_func(loss_mask, output):
    result_dict_ = {}
    if isinstance(output, tuple):
        # pydevd_pycharm.settrace("localhost", port=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)
        lm_loss, result_dict_ = output
    else:
        lm_loss = output

    lm_loss_ = lm_loss.float()
    lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    loss = lm_loss
    averaged_losses = average_losses_across_data_parallel_group([lm_loss])
    result_dict = {"CE loss": averaged_losses[0]}
    result_dict.update(result_dict_)
    return loss, result_dict


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers("batch generator", log_level=2).start()
    docs_enc, sent_nums, labels, loss_mask, enc_mask, sent_mask, enc_dec_mask, dec_mask = get_batch(data_iterator)
    timers("batch generator").stop()

    # Forward model lm_labels
    output_tensor = model(
        docs_enc,
        labels,
        sent_nums,
        sent_mask=sent_mask,
        enc_mask=enc_mask,
        enc_dec_mask=enc_dec_mask,
        dec_mask=dec_mask,
        tokentype_ids=None,
        lm_labels=labels,
    )

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0("> building train, validation, and test datasets " "for CodeFormer ...")
    data_prefix=args.data_path
    label_prefix = [prefix.replace("code", "label") for prefix in data_prefix]
    # pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=data_prefix,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.max_sent_length,
        max_seq_length_dec=args.decoder_seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        dataset_type="codeformer",
        label_prefix=label_prefix
    )
    print_rank_0("> finished creating CodeFormer datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_and_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "HFTokenizer", "tokenizer_model": "Salesforce/codet5p-220m"},
    )
