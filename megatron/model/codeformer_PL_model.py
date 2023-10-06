# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Transformer based language model."""

import torch
from einops import rearrange

from megatron import get_args
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.common.rotary_pos_embedding import RotaryEmbedding

from .enums import AttnMaskType, LayerType
from .module import MegatronModule
from .transformer import ParallelTransformer
from .utils import get_linear_layer
from .utils import init_method_normal, scaled_init_method_normal

import pydevd_pycharm


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output, bias=None):
    """LM logits using word embedding weights."""
    args = get_args()
    # Parallel logits.
    if args.async_tensor_model_parallel_allreduce or args.sequence_parallel:
        input_parallel = input_
        model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        async_grad_allreduce = (
            args.async_tensor_model_parallel_allreduce and model_parallel and not args.sequence_parallel
        )
    else:
        input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(input_)
        async_grad_allreduce = False

    # Matrix multiply.
    logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
        input=input_parallel,
        weight=word_embeddings_weight,
        bias=bias,
        gradient_accumulation_fusion=args.gradient_accumulation_fusion,
        async_grad_allreduce=async_grad_allreduce,
        sequence_parallel=args.sequence_parallel,
    )
    # Gather if needed.

    if parallel_output:
        return logits_parallel

    return tensor_parallel.gather_from_tensor_model_parallel_region(logits_parallel)


def get_language_model(
    config,
    add_pooler,
    add_encoder=True,
    add_decoder=True,
    decoder_attn_mask_type=AttnMaskType.causal,
    pre_process=True,
    post_process=True,
):
    """Build language model and return along with the key to save."""
    args = get_args()
    if config.init_method is None:
        config.init_method = init_method_normal(config.init_method_std)

    if config.output_layer_init_method is None:
        config.output_layer_init_method = scaled_init_method_normal(config.init_method_std, config.num_layers)

    # Language model.
    language_model = СodeformerLanguageModel(
        config,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process,
    )
    # key used for checkpoints.
    language_model_key = "codeformer_model"

    return language_model, language_model_key


class Pooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method):
        super(Pooler, self).__init__()
        args = get_args()
        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)
        self.sequence_parallel = args.sequence_parallel

    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [s, b, h]
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False
            )

        pooled = hidden_states[sequence_index, :, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
        embedding_weights_in_fp32: casts word embedding weights to
                                   fp32 before sampling. Required to
                                   maintain reproducibility when
                                   training in bf16.
    """

    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        config,
        num_tokentypes=0,
        embedding_weights_in_fp32=False,
    ):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = config.init_method
        self.num_tokentypes = num_tokentypes

        args = get_args()

        # Word embeddings (parallel).
        self.embedding_weights_in_fp32 = embedding_weights_in_fp32
        self.params_dtype = args.params_dtype
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            vocab_size, self.hidden_size, config=config, init_method=config.init_method
        )
        self._word_embeddings_key = "word_embeddings"

        # Position embedding (serial).
        self.add_position_embedding = args.position_embedding_type == "learned_absolute"
        if self.add_position_embedding:
            self.position_embeddings = torch.nn.Embedding(max_sequence_length, self.hidden_size)
            self._position_embeddings_key = "position_embeddings"
            # Initialize the position embeddings.
            if args.perform_initialization:
                self.init_method(self.position_embeddings.weight)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = "tokentype_embeddings"
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes, self.hidden_size)
            # Initialize the token-type embeddings.
            if args.perform_initialization:
                self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        self.fp32_residual_connection = args.fp32_residual_connection
        self.sequence_parallel = args.sequence_parallel
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        if self.add_position_embedding:
            self.position_embeddings.weight.data.fill_(0)
            self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception("tokentype embeddings is already initialized")
        if torch.distributed.get_rank() == 0:
            print("adding embedding for {} tokentypes".format(num_tokentypes), flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes, self.hidden_size)
        # Initialize the token-type embeddings.
        args = get_args()
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # Embeddings.
        if self.embedding_weights_in_fp32:
            self.word_embeddings = self.word_embeddings.to(torch.float32)
        words_embeddings = self.word_embeddings(input_ids)
        if self.embedding_weights_in_fp32:
            words_embeddings = words_embeddings.to(self.params_dtype)
            self.word_embeddings = self.word_embeddings.to(self.params_dtype)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] = self.word_embeddings.state_dict(prefix=prefix, keep_vars=keep_vars)
        if self.add_position_embedding:
            state_dict_[self._position_embeddings_key] = self.position_embeddings.state_dict(
                prefix=prefix, keep_vars=keep_vars
            )
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] = self.tokentype_embeddings.state_dict(
                prefix=prefix, keep_vars=keep_vars
            )

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if "word_embeddings" in key:
                    state_dict_[key.split("word_embeddings.")[1]] = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self.add_position_embedding:
            if self._position_embeddings_key in state_dict:
                state_dict_ = state_dict[self._position_embeddings_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if "position_embeddings" in key:
                        state_dict_[key.split("position_embeddings.")[1]] = state_dict[key]
            self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if "tokentype_embeddings" in key:
                        state_dict_[key.split("tokentype_embeddings.")[1]] = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_, strict=strict)
            else:
                print(
                    "***WARNING*** expected tokentype embeddings in the " "checkpoint but could not find it", flush=True
                )


class СodeformerLanguageModel(MegatronModule):
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
        super(СodeformerLanguageModel, self).__init__(
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

            ## TODO DISCUSS decide shouldn't we use shared emb
            self.embedding_dec = Embedding(
                hidden_size=self.hidden_size,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                embedding_dropout_prob=args.hidden_dropout,
                config=config,
                num_tokentypes=self.num_tokentypes,
                embedding_weights_in_fp32=args.embedding_weights_in_fp32,
            )
            self._embedding_dec_key = "embedding_dec"

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
        # pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)
        # Encoder (usually set to True, False if part of an encoder-decoder
        # architecture and in encoder-only stage).
        self.encoder_1 = ParallelTransformer(
            config,
            model_type=ModelType.encoder_and_decoder,
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
            is_codeformer=True,
            enc_num=2,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        self._encoder_2_key = "encoder_2"

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
        # TODO input_tensor is always None. When does it change?
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

        # pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)
        # Encoder embedding.
        b = enc_input_ids.size(0)
        max_sent_num_batch = sent_mask.size(2)
        pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)
        enc_input_ids = rearrange(enc_input_ids, "b (s t) -> (b s) t", t=self.max_sent_length)
        enc_mask = rearrange(enc_mask, "b 1 s ... -> (b s) 1 ...")

        enc_position_ids = self.get_position_ids(enc_input_ids)
        dec_position_ids = self.get_position_ids(dec_input_ids)
        if self.pre_process:
            encoder_input = self.embedding(enc_input_ids, enc_position_ids, tokentype_ids=tokentype_ids)
            decoder_input = self.embedding_dec(dec_input_ids, dec_position_ids, tokentype_ids=tokentype_ids)
        else:
            encoder_input = None

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
    
    # dec_mask,
        # enc_dec_mask,
        # sent_mask,

        # Run encoder.
        if enc_hidden_states is None:
            encoder_output = self.encoder_1(
                encoder_input,
                attention_mask=enc_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb_1,
            )

            encoder_output = rearrange(encoder_output, "t (b s) d -> s b d t", b=b)
            encoder_output = self.linear(encoder_output).squeeze(-1)

            encoder_output = self.encoder_2(
                encoder_output,
                attention_mask=sent_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb_2,
            )

            output = self.decoder(
                decoder_input,
                attention_mask=dec_mask,
                encoder_output=encoder_output,
                enc_dec_attn_mask=enc_dec_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb_dec,
            )

        else:
            output = enc_hidden_states.to(encoder_input.dtype)

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

            state_dict_[self._embedding_dec_key] = self.embedding_dec.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars
            )
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
            assert self._embedding_dec_key in state_dict, "No embedding for decoder weights in the checkpoint!"
            state_dict_ = state_dict[self._embedding_key]
            state_dict_dec_ = state_dict[self._embedding_dec_key]
            self.embedding.load_state_dict(state_dict_, strict=strict)
            self.embedding_dec.load_state_dict(state_dict_dec_, strict=strict)

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
