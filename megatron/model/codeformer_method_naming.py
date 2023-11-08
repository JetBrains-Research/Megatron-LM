# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Transformer based language model."""

import torch

from megatron import get_args
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.common.rotary_pos_embedding import RotaryEmbedding
from megatron.model.codeformer_common_functions import reshape_aggregated, Embedding, Pooler

from .enums import AttnMaskType, LayerType
from .module import MegatronModule
from .transformer import ParallelTransformer
from .utils import get_linear_layer

import pydevd_pycharm

class СodeformerMethodNaming(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        config,
        # encoder_attn_mask_type,
        add_encoder=True,
        add_decoder=False,
        decoder_attn_mask_type=AttnMaskType.causal,
        add_pooler=False,
        pre_process=True,
        post_process=True,
        num_tokentypes=0,
    ):
        args = get_args()
        # TODO: passing share_embeddings_and_output_weights=False will not work correctly for T5 and embeddings will not be synced. Fix later for T5.
        super(СodeformerMethodNaming, self).__init__(
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = config.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = config.init_method
        # self.encoder_attn_mask_type = encoder_attn_mask_type
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.encoder_hidden_state = None
        self.add_retriever = args.retro_add_retriever
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.max_sent_length = args.max_sent_length + 2
        self.max_sent_num = args.max_sent_num
        self.max_label_length = args.max_label_length + 2

        # Embeddings.
        if self.pre_process:
            self.embedding = Embedding(
                hidden_size=self.hidden_size,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                embedding_dropout_prob=args.hidden_dropout,
                config=config,
                num_tokentypes=self.num_tokentypes,
                embedding_weights_in_fp32=args.embedding_weights_in_fp32,
            )

            self._embedding_key = "embedding"

            # ## TODO DISCUSS decide shouldn't we use shared emb
            # self.embedding_dec = Embedding(
            #     hidden_size=self.hidden_size,
            #     vocab_size=args.padded_vocab_size,
            #     max_sequence_length=args.max_position_embeddings,
            #     embedding_dropout_prob=args.hidden_dropout,
            #     config=config,
            #     num_tokentypes=self.num_tokentypes,
            #     embedding_weights_in_fp32=args.embedding_weights_in_fp32,
            # )
            # self._embedding_dec_key = "embedding_dec"

        # Rotary positional embeddings
        self.use_rotary_position_embeddings = args.position_embedding_type == "rope"
        if self.use_rotary_position_embeddings:
            rotary_dim = args.hidden_size // args.num_attention_heads if args.kv_channels is None else args.kv_channels

            if args.rotary_percent < 1.0:
                rotary_dim = int(rotary_dim * args.rotary_percent)

            # partial rotary embeddings, which is better than full rotary
            # Wang and Komatsuzaki et al
            # https://github.com/kingoflolz/mesh-transformer-jax/
            # TODO CHECK that we need three RoPE instances
            self.rotary_pos_emb_1 = RotaryEmbedding(
                rotary_dim, seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
            )

            self.rotary_pos_emb_2 = RotaryEmbedding(
                rotary_dim, seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
            )

            self.rotary_pos_emb_dec = RotaryEmbedding(
                rotary_dim, seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
            )

        # Encoder (usually set to True, False if part of an encoder-decoder
        # architecture and in encoder-only stage).
        self.encoder_1 = ParallelTransformer(
            config,
            model_type=ModelType.encoder_and_decoder,
            self_attn_mask_type = AttnMaskType.padding,
            is_codeformer=True,
            enc_num=1,
            pre_process=self.pre_process,
            post_process=self.post_process,  # If postprocess, then LayerNorm is applied to the output
        )
        self._encoder_1_key = "encoder_1"

        # args.max_position_embeddings = args.max_sent_length+2 due toa dditinal BOS and EOS tokens
        self.linear = get_linear_layer(args.max_position_embeddings, 1, config.init_method)
        self._linear_key = "linear"

        self.encoder_2 = ParallelTransformer(
            config,
            model_type=ModelType.encoder_and_decoder,
            self_attn_mask_type = AttnMaskType.padding,
            is_codeformer=True,
            enc_num=2,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        self._encoder_2_key = "encoder_2"

        # for p1, p2 in zip(self.encoder_1.parameters(), self.encoder_2.parameters()):
        #     p2.data = p1.data

        self.decoder = ParallelTransformer(
            config,
            model_type=ModelType.encoder_and_decoder,
            is_codeformer=True,
            layer_type=LayerType.decoder,
            self_attn_mask_type=self.decoder_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        self._decoder_key = "decoder"

        if self.post_process:
            # Pooler.
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method)
                self._pooler_key = "pooler"

            if self.untie_embeddings_and_output_weights:
                self.output_layer = tensor_parallel.ColumnParallelLinear(
                    args.hidden_size, args.padded_vocab_size, config=config, init_method=self.init_method, bias=False
                )  # Setting bias to False always to keep it consistent with embedding tying that also does not have a bias.
                self._output_layer_key = "output_layer"

    def get_position_ids(self, token_ids):
        # Create position ids
        seq_length = token_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

        return position_ids

    def set_input_tensor(self, input_tensor):

        """See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        # TODO input_tensor is always None. When does it change? Model parallelism?
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
            # if input_tensor is not None:
            #     import pydevd_pycharm
            #     pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)

        self.encoder_1.set_input_tensor(input_tensor[0])

        if self.add_encoder and self.add_decoder:
            assert (
                len(input_tensor) == 1
            ), "input_tensor should only be length 1 for stage with both encoder and decoder"
            self.encoder_1.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            assert len(input_tensor) == 1, "input_tensor should only be length 1 for stage with only encoder"
            self.encoder_1.set_input_tensor(input_tensor[0])
        elif self.add_decoder:
            if len(input_tensor) == 2:
                self.decoder.set_input_tensor(input_tensor[0])
                self.encoder_hidden_state = input_tensor[1]
            elif len(input_tensor) == 1:
                self.decoder.set_input_tensor(None)
                self.encoder_hidden_state = input_tensor[0]
            else:
                raise Exception("input_tensor must have either length 1 or 2")
        else:
            raise Exception("Stage must have at least either encoder or decoder")

        assert len(input_tensor) == 1, "input_tensor should only be length 1 for stage with only encoder"

    def forward(
        self,
        enc_input_ids,
        dec_input_ids,
        sent_nums,
        enc_mask,
        sent_mask,
        enc_dec_mask,
        dec_mask,
        tokentype_ids=None,
        inference_params=None,
        pooling_sequence_index=0,
        enc_hidden_states=None,
        output_enc_hidden=False,
    ):

        # Encoder embedding.
        b = sent_nums.size(0)
        max_sent_num_batch = sent_mask.size(2)

        enc_position_ids = self.get_position_ids(enc_input_ids)
        dec_position_ids = self.get_position_ids(dec_input_ids)
        if self.pre_process:
            input_embeddings = self.embedding(enc_input_ids, enc_position_ids, tokentype_ids=tokentype_ids)
            # decoder_input = self.embedding_dec(dec_input_ids, dec_position_ids, tokentype_ids=tokentype_ids)
            decoder_input = self.embedding(dec_input_ids, dec_position_ids, tokentype_ids=tokentype_ids)
        else:
            input_embeddings = None

        # Rotary positional embeddings
        rotary_pos_emb = None
        if self.use_rotary_position_embeddings:
            if inference_params is not None:
                # TODO FINAL may be for inference something should be changed
                # rotary_pos_emb = self.rotary_pos_emb(inference_params.max_sequence_length)
                rotary_pos_emb_1 = self.rotary_pos_emb_1(self.max_sent_length)
                rotary_pos_emb_2 = self.rotary_pos_emb_2(max_sent_num_batch)
                rotary_pos_emb_dec = self.rotary_pos_emb_dec(self.max_label_length)
            else:
                rotary_pos_emb_1 = self.rotary_pos_emb_1(self.max_sent_length)
                rotary_pos_emb_2 = self.rotary_pos_emb_2(max_sent_num_batch)
                rotary_pos_emb_dec = self.rotary_pos_emb_dec(self.max_label_length)
        # pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)
        if enc_hidden_states is None:
            input_embeddings = self.encoder_1(
                input_embeddings,
                attention_mask=enc_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb_1,
            )

            # input_embeddings = rearrange(input_embeddings, "t (b s) d -> s b d t", b=b)
            input_embeddings = input_embeddings.permute(1, 2, 0) # t (b s) d -> (b s) d t
            input_embeddings = self.linear(input_embeddings).squeeze(-1) # (b s) d t -> (b s) d

            # reshaping embeddings to batches (b s) d -> s_max b d
            encoder_input = reshape_aggregated(input_embeddings, sent_nums, max_sent_num_batch, b)

            input_embeddings = self.encoder_2(
                encoder_input,
                attention_mask=sent_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb_2,
            )

            # pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)
            # decoder_input = torch.concat((input_embeddings[7:14], decoder_input))
            # dec_mask_new = torch.BoolTensor(b,1,14,14).fill_(False).to(dec_mask.device)
            # dec_mask_new[:,:,:7,7:] = True
            # dec_mask_new[:, :, 7:, 7:] = dec_mask
            #
            #
            # enc_dec_mask_new = torch.BoolTensor(b,1,14,enc_dec_mask.shape[-1]).fill_(False).to(enc_dec_mask.device)
            # enc_dec_mask_new[:, :, 7:, :] = enc_dec_mask

            output = self.decoder(
                decoder_input,
                attention_mask=dec_mask,
                encoder_output=input_embeddings,
                enc_dec_attn_mask=enc_dec_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb_dec,
            )
            # output = output[7:]
        else:
            output = enc_hidden_states.to(input_embeddings.dtype)

        # output_enc_hidden refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        return output

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """For easy load."""
        # print("------------------------------------")
        # print("--------- Saving model -------------")
        # print("------------------------------------")
        state_dict_ = {}
        if self.pre_process:
            state_dict_[self._embedding_key] = self.embedding.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars
            )

            # state_dict_[self._embedding_dec_key] = self.embedding_dec.state_dict_for_save_checkpoint(
            #     prefix=prefix, keep_vars=keep_vars
            # )
        state_dict_[self._encoder_1_key] = self.encoder_1.state_dict_for_save_checkpoint(
            prefix=prefix, keep_vars=keep_vars
        )
        state_dict_[self._linear_key] = self.linear.state_dict(
            prefix=prefix, keep_vars=keep_vars
        )
        state_dict_[self._encoder_2_key] = self.encoder_2.state_dict_for_save_checkpoint(
            prefix=prefix, keep_vars=keep_vars
        )
        if self.post_process:
            if self.add_pooler:
                state_dict_[self._pooler_key] = self.pooler.state_dict_for_save_checkpoint(
                    prefix=prefix, keep_vars=keep_vars
                )
            if self.untie_embeddings_and_output_weights:
                state_dict_[self._output_layer_key] = self.output_layer.state_dict(prefix=prefix, keep_vars=keep_vars)

        state_dict_[self._decoder_key] = self.decoder.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # print("------------------------------------")
        # print("--------- Loading model -------------")
        # print("------------------------------------")

        # Embedding.
        if self.pre_process:
            assert self._embedding_key in state_dict, "No embedding weights in the checkpoint!"
            # assert self._embedding_dec_key in state_dict, "No embedding for decoder weights in the checkpoint!"
            state_dict_ = state_dict[self._embedding_key]
            # state_dict_dec_ = state_dict[self._embedding_dec_key]
            self.embedding.load_state_dict(state_dict_, strict=strict)
            # self.embedding_dec.load_state_dict(state_dict_dec_, strict=strict)

        # Encoder.
        if self.add_encoder:
            assert self._encoder_1_key in state_dict, "No encoder 1 weights in the checkpoint!"
            assert self._encoder_2_key in state_dict, "No encoder 2 weights in the checkpoint!"
            state_dict_1_ = state_dict[self._encoder_1_key]
            state_dict_2_ = state_dict[self._encoder_2_key]
            self.encoder_1.load_state_dict(state_dict_1_, strict=strict)
            self.encoder_2.load_state_dict(state_dict_2_, strict=strict)

        assert "linear" in state_dict, "No linear weights in the checkpoint"
        self.linear.load_state_dict(state_dict[self._linear_key], strict=strict)

        # Pooler.
        if self.post_process:
            if self.add_pooler:
                assert "pooler" in state_dict, "could not find data for pooler in the checkpoint"
                self.pooler.load_state_dict(state_dict[self._pooler_key], strict=strict)
            if self.untie_embeddings_and_output_weights:
                assert "output_layer" in state_dict, "could not find data for output_layer in the checkpoint"
                self.output_layer.load_state_dict(state_dict[self._output_layer_key], strict=strict)
        # Decoder.
        if self.add_decoder:
            assert "decoder" in state_dict, "No decoder weights in the checkpoint"
            self.decoder.load_state_dict(state_dict[self._decoder_key], strict=strict)
