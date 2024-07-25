#!/usr/bin/env python3

import logging
import math, sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask, lengths_to_mask
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerEncoderLayer
from fairseq.models.wav2vec import Wav2Vec2Model
from fairseq.models.speech_to_text.fast_pretraining import FASTPretrainingModel, FASTPretrainingEncoder, TransformerDecoderNoExtra, base_architecture

logger = logging.getLogger(__name__)
torch.set_printoptions(threshold=10_000)


THRESHOLD = 0.999
@register_model("fast_offline")
class FASTOfflineModel(FASTPretrainingModel):
    """
    Transformer-based Speech translation model from ESPNet-ST
    https://arxiv.org/abs/2004.10234
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(FASTOfflineModel, FASTOfflineModel).add_args(parser)
        parser.add_argument(
            "--fast-cif",
            default=True,
            type=bool,
            help="if True, use the fast cif function",
        )
        parser.add_argument(
            "--simul-mode",
            type=str,
            choices=["vanilla", "fine-mask", "fine-wait"],
            help="if True, perform streaming inference",
        )
        parser.add_argument(
            "--load-pretrained-mt-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )

    @classmethod
    def build_wav2vec_model(cls, args):
        wav2vec_ckpt = torch.load(args.w2v2_model_path)

        wav2vec_ckpt['args'].simul_mode = args.simul_mode
        wav2vec_ckpt['args'].future_num = args.future_num

        if getattr(wav2vec_ckpt['args'], "w2v_args", None) is None:

            wav2vec_model = Wav2Vec2Model.build_model(wav2vec_ckpt['args'], task=None)
            wav2vec_model.load_state_dict(wav2vec_ckpt['model'])
        else:
            wav2vec_model = Wav2Vec2Model.build_model(wav2vec_ckpt['args'].w2v_args, task=None)
            w2v_asr_states = {}

            for k, v in wav2vec_ckpt['model'].items():
                if "w2v_encoder.w2v_model." in k:
                    w2v_asr_states[k.replace("w2v_encoder.w2v_model.", "")] = v

            with torch.no_grad():
                wav2vec_model.load_state_dict(w2v_asr_states, strict=False)

        logger.info(f"loaded pretrained wav2vec 2.0 from: {args.w2v2_model_path}")
        
        return wav2vec_model

    @classmethod
    def build_encoder(cls, args):
        wav2vec_model = cls.build_wav2vec_model(args)

        encoder = FASTOfflineEncoder(args, wav2vec_model)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=pretraining_path, strict=False
            )
            logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        
        pretraining_mt_path = getattr(args, "load_pretrained_mt_encoder_from", None)

        if pretraining_mt_path is not None:
            transformer_layers = encoder.transformer_layers

            mt_states = torch.load(pretraining_mt_path)["model"]
            layers_states = {}

            for k, v in mt_states.items():
                if "encoder.layers." in k:
                    layers_states[k.replace("encoder.layers.", "")] = v

            with torch.no_grad():
                transformer_layers.load_state_dict(layers_states)
            logger.info(f"loaded transformer_layers from mt model")

        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = TransformerDecoderNoExtra(args, task.target_dictionary, embed_tokens)
        pretraining_path = getattr(args, "load_pretrained_decoder_from", None)
        if pretraining_path is not None:
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=pretraining_path, strict=False
            )
            logger.info(f"loaded pretrained decoder from: {pretraining_path}")
                
        return decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens, target_lengths=None):

        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, target_lengths=target_lengths)
        
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )

        return {'logits': decoder_out, 'len_logits': target_lengths,
                'alphas': encoder_out["alphas"], 'num_output': encoder_out["num_output"]}


class FASTOfflineEncoder(FASTPretrainingEncoder):
    """Conv + Transformer encoder"""

    def __init__(self, args, wav2vec_model):
        """Construct an Encoder object."""
        super().__init__(args, wav2vec_model)

        self.simul_mode = args.simul_mode
        self.cif = self.fast_cif if args.fast_cif else self.original_cif

        self.proj = self.Linear(args.encoder_embed_dim - 1, args.encoder_embed_dim)

    def _get_w2v_feature(self, src_tokens, src_lengths, finish_read=True):
        """
        :param src_tokens: b x frames
        :param src_lengths: b-dim length
        :return: w2v_feature: b x short_frames x feature-dim;
                w2v_lengths: b-dim tensor
                w2v_padding_mask: b x short_frames x feature-dim T/F tensor
        """
        padding_mask = lengths_to_padding_mask(src_lengths)
        w2v_res = self.wav2vec_model.extract_features(src_tokens, padding_mask, finish_read=finish_read)
        return w2v_res

    def get_alphas(self, encoder_output, padding_mask):
        alphas = encoder_output[:, :, -1]
        alphas = torch.sigmoid(alphas)
        alphas = alphas * (~padding_mask).float()  #fp32
        return alphas

    def original_cif(self, encoder_output, alphas, threshold=THRESHOLD, supervised_lengths=None):
        """
        supervised_lengths is unused args, just for code compatiablity with fast_cif.

        This code is slow and buggy.
        e.g., 
           if the target length is L, where L is an integer.
           and the alphas.sum() = (L - 1) + F, 
               where F is a float close to 1.0 but smaller than threshold=0.999 
               e.g., F = 0.997
           According to the follwong code snippet,
              ```
              l = torch.index_select(frames[b, :, :], 0, torch.where(fire >= threshold)[0])
              pad_l = torch.zeros([max_label_len - l.size(0), H], device=device)
              list_ls.append(torch.cat([l, pad_l], 0))
              ``` 
           the actual length is (L - 1) and position L will be padding as value 0.
        """
        if type(encoder_output) is Tensor:
            hidden = encoder_output
        elif 'encoded' in encoder_output.keys():
            hidden = encoder_output['encoded'][0][:, :, :-1]
        else:
            hidden = encoder_output['encoder_out'][0][:, :, :-1]

        device = hidden.device
        B, T, H = hidden.size()

        # loop varss
        integrate = torch.zeros([B], device=device)  #fp32
        frame = torch.zeros([B, H], device=device)   #fp32
        # intermediate vars along time
        list_fires = []
        list_frames = []

        for t in range(T):
            alpha = alphas[:, t]
            distribution_completion = 1 - integrate

            integrate += alpha
            list_fires.append(integrate)

            fire_place = integrate >= threshold
            integrate = torch.where(fire_place,
                                    integrate - 1,
                                    integrate)
            cur = torch.where(fire_place,
                              distribution_completion,
                              alpha)
            remainds = alpha - cur

            frame += cur[:, None] * hidden[:, t, :]
            list_frames.append(frame)
            frame = torch.where(fire_place[:, None].repeat(1, H),
                                remainds[:, None] * hidden[:, t, :],
                                frame)

        fires = torch.stack(list_fires, 1)
        frames = torch.stack(list_frames, 1)
        list_ls = []
        len_labels = torch.round(alphas.sum(-1)).int()
        len_labels[len_labels < 1] = 1
        max_label_len = len_labels.max()
        for b in range(B):
            fire = fires[b, :]
            l = torch.index_select(frames[b, :, :], 0, torch.where(fire >= threshold)[0])
            pad_l = torch.zeros([max_label_len - l.size(0), H], device=device)
            list_ls.append(torch.cat([l, pad_l], 0))

        return torch.stack(list_ls, 0)

    def fast_cif(self, encoder_output, alphas, threshold=THRESHOLD, supervised_lengths=None):
        """
        because during kd training or evaluation, the sum of partial alphas (teacher) could be 
        a number representing as integer + decimal.
        The decimal could be < 0.5 and signficantly > 0.0, e.g., 7.3
        The len_labels will be 7 after rounding.
        
        In the orginal cif code, all decimal will be discarded.
        However, in the fast cif code, remainds such as 0.3 won't be omitted.
        The actual length will be 8, and the weights will have index out of bound ERROR.
        So I will put the remainds portion to 7 rather 8.
        If the remainds > 0.5, the remainds will be 8.
        This straregy matches the len_labels perfectly, i.e., matching round().
        """
        if type(encoder_output) is Tensor:
            hidden = encoder_output
        elif 'encoded' in encoder_output.keys():
            hidden = encoder_output['encoded'][0][:, :, :-1]
        else:
            hidden = encoder_output['encoder_out'][0][:, :, :-1]

        device = hidden.device
        B, T, H = hidden.size()
   
        if supervised_lengths is not None:
            len_labels = supervised_lengths
        else:
            len_labels = torch.round(alphas.sum(-1)).int()
        len_labels[len_labels < 1] = 1
        max_label_len = len_labels.max()

        # loop vars
        integrate = torch.zeros([B], dtype=alphas.dtype, device=device)
        remainds = torch.zeros([B], dtype=alphas.dtype, device=device)
        fire_num = torch.zeros([B], dtype=torch.long, device=device)

        weights = torch.zeros((B, max_label_len, T), dtype=alphas.dtype, device=device)
        for t in range(T):
            if t > 0:
                weights[:, :, t - 1].scatter_add_(dim=1, index=fire_num.unsqueeze(1), src=remainds.unsqueeze(1))
        
            alpha = alphas[:, t]
            distribution_completion = 1 - integrate
            integrate += alpha
            fire_place = integrate >= threshold

            integrate = torch.where(fire_place, integrate - 1, integrate)
            cur = torch.where(fire_place, distribution_completion, alpha)

            weights[:, :, t].scatter_(dim=1, index=fire_num.unsqueeze(1), src=cur.unsqueeze(1))
            remainds = alpha - cur
       
            fire_num = fire_num + fire_place.type_as(fire_num)
            fire_num = torch.minimum(fire_num, len_labels - 1)

        return weights.bmm(hidden.type_as(weights))

    def resize(self, alphas, target_lengths, noise=0.0, threshold=THRESHOLD):
        """
        alpha in thresh=1.0 | (0.0, +0.21)
        target_lengths: if None, apply round and resize, else apply scaling
        """
        device = alphas.device
        # sum
        _num = alphas.sum(-1)

        num = target_lengths.type_as(_num)
        num = num + noise * torch.rand(alphas.size(0)).to(device)

        # scaling
        _alphas = alphas * (num / _num).unsqueeze(-1)

        # rm attention value that exceeds threashold
        count = 0
        while len(torch.where(_alphas > threshold)[0]):
            count += 1
            if count > 10:
                break
            # print('fixing alpha')
            xs, ys = torch.where(_alphas > threshold)
            for x, y in zip(xs, ys):
                if _alphas[x][y] >= threshold:
                    mask = _alphas[x].ne(0).type_as(_alphas)
                    mean = 0.5 * _alphas[x].sum() / mask.sum()
                    _alphas[x] = _alphas[x] * 0.5 + mean * mask

        return _alphas, _num

    def add_positions(self, x, encoder_padding_mask):
        positions = self.embed_positions(encoder_padding_mask)

        x = x + positions
        # The last dim cannot be dropout because it will be used for CIF.
        F.dropout(x[:, :, :-1], p=self.dropout, training=self.training, inplace=True)
        return x

    def set_dtype(self, x, cif_outputs):
        _x_type = x.dtype
        _proj_type = self.proj.weight.dtype
        cif_outputs = cif_outputs.type(_proj_type)

        x = self.proj(cif_outputs)
        x = x.type(_x_type)
        
        return x

    def forward(self, src_tokens, src_lengths, target_lengths=None, finish_read=True):
        """Encode input sequence.
        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        w2v_res = self._get_w2v_feature(src_tokens, src_lengths, finish_read=finish_read)
        w2v_feature = w2v_res["x"]
        encoder_padding_mask = w2v_res["padding_mask"]

        x = w2v_feature 
        
        # add position embedding
        x = self.add_positions(x, encoder_padding_mask)

        # cif
        # calculate alphas
        alphas = self.get_alphas(x, encoder_padding_mask)

        if self.training:
            decode_length = target_lengths
            noise = 0.0
        else:
            decode_length = torch.round(alphas.sum(-1)).int()
            decode_length[decode_length==0] = 1
            noise = 0.0

        # resize alphas
        _alphas, num_output = self.resize(alphas, decode_length, noise=noise)
        padding_mask = lengths_to_padding_mask(decode_length)

        # weight average for compressing output
        if self.training:
            cif_outputs = self.cif(x[:, :, :-1], _alphas, supervised_lengths=target_lengths)
        else:
            cif_outputs = self.cif(x[:, :, :-1], _alphas)

        x = self.set_dtype(x, cif_outputs)
        # finished add cif

        # x = self.layer_norm(x)

        x = torch.transpose(x, 1, 0)

        # Semantic Encoder
        for layer in self.transformer_layers:
            x = layer(x, padding_mask)

        maybe_encoder_padding_mask = encoder_padding_mask

        return {
            "encoder_out": [x],
            "encoder_padding_mask": [padding_mask] if maybe_encoder_padding_mask is not None else [],
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
            "alphas": alphas,
            "num_output": num_output,
        }


@register_model_architecture("fast_offline", "fast_offline")
def fast_offline(args):
    args.simul_mode = getattr(args, "simul_mode", None)
    args.future_num = getattr(args, "future_num", 0)
