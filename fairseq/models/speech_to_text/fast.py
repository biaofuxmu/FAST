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
from fairseq.models.transformer import Embedding
from fairseq.modules import LayerNorm
from fairseq.models.wav2vec import Wav2Vec2KDModel
from fairseq.models.speech_to_text.fast_offline import FASTOfflineModel, FASTOfflineEncoder
from fairseq.models.speech_to_text.fast_pretraining import base_architecture

logger = logging.getLogger(__name__)
torch.set_printoptions(threshold=10_000)


@register_model("fast")
class FASTModel(FASTOfflineModel):
    """
    Transformer-based Speech translation model from ESPNet-ST
    https://arxiv.org/abs/2004.10234
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(FASTModel, FASTModel).add_args(parser)
        parser.add_argument(
            "--future-mask-length",
            type=int,
            default=50,
            help="future context speech frame length",
        )
        parser.add_argument(
            "--kd-loss",
             type=str,
             choices=["cos", "mse"],
             help="kd loss type for hidden states.",
        )


    @classmethod
    def build_wav2vec_model(cls, args):
        wav2vec_ckpt = torch.load(args.w2v2_model_path)

        wav2vec_ckpt['args'].simul_mode = args.simul_mode
        wav2vec_ckpt['args'].future_num = args.future_num
        wav2vec_ckpt['args'].future_mask_length = args.future_mask_length

        wav2vec_model = Wav2Vec2KDModel.build_model(wav2vec_ckpt['args'], task=None)
        wav2vec_model.load_state_dict(wav2vec_ckpt['model'], strict=False)

        return wav2vec_model

    @classmethod
    def build_encoder(cls, args):
        wav2vec_model = cls.build_wav2vec_model(args)

        encoder = FASTEncoder(args, wav2vec_model)

        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=pretraining_path, strict=False
            )
            logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        if cls.simul_mode is None:
            # init student encoder parameters from teacher encoder
            teacher_w2v2_encoder = encoder.wav2vec_model.encoder
            student_w2v2_encoder = encoder.wav2vec_model.stu_encoder
            with torch.no_grad():
                student_w2v2_encoder.load_state_dict(teacher_w2v2_encoder.state_dict())
            logger.info(f"loaded student w2v2 encoder from teacher model")


        return encoder
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        cls.simul_mode = args.simul_mode
        
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )

        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)

        if cls.simul_mode is None:
            # frozen teacher model
            for name, param in encoder.named_parameters():
                if "stu" not in name:
                    param.requires_grad = False

            for _, param in decoder.named_parameters():
                param.requires_grad = False

        return cls(encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, target_lengths=None):

        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, target_lengths=target_lengths)

        if self.simul_mode is None:
            return {'len_logits': target_lengths,
                    'alphas': encoder_out["alphas"], 'num_output': encoder_out["num_output"],
                    "w2v2_kd_loss": encoder_out["w2v2_kd_loss"],
                    "cif_kd_loss": encoder_out["cif_kd_loss"],
                    }

        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )

        return {'logits': decoder_out, 'len_logits': target_lengths,
                'alphas': encoder_out["alphas"], 'num_output': encoder_out["num_output"]}


class FASTEncoder(FASTOfflineEncoder):
    """Conv + Transformer encoder"""

    def __init__(self, args, wav2vec_model):
        """Construct an Encoder object."""
        super().__init__(args, wav2vec_model)

        if self.simul_mode is None:
            if args.kd_loss == "cos":
                self.kd_loss_func = cosine_distance_loss
            elif args.kd_loss == "mse":
                self.kd_loss_func = token_level_mse_loss
            else:
                raise ValueError("not defined kd loss: %s." % args.kd_loss)


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

        if self.simul_mode is None:
            mask_w2v2_output = w2v_res["mask_output"]
            wait_w2v2_output = w2v_res["wait_output"]
            sampled_lengths_mask = w2v_res["sampled_lengths_mask"]

            # w2v2 kd loss
            non_padding_mask = sampled_lengths_mask.type_as(mask_w2v2_output)
            w2v2_kd_loss = self.kd_loss_func(
                mask_w2v2_output, wait_w2v2_output, non_padding_mask)

            # cif_kd_loss
            cif_kd_loss = self.kd_output(mask_w2v2_output, wait_w2v2_output, sampled_lengths_mask)

            return {
                "encoder_out": [x],
                "encoder_padding_mask": [],
                "encoder_embedding": [],
                "encoder_states": [],
                "src_tokens": [],
                "src_lengths": [],
                "alphas": None,
                "num_output": None,
                "w2v2_kd_loss": w2v2_kd_loss,
                "cif_kd_loss": cif_kd_loss,
            }

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

    def kd_output(self, mask_w2v2_output, wait_w2v2_output, sampled_lengths_mask):
        # add positions to mask and wait output

        mask_output = self.add_positions(mask_w2v2_output.clone(), ~sampled_lengths_mask)
        wait_output = self.add_positions(wait_w2v2_output.clone(), ~sampled_lengths_mask)

        # cif kd loss
        mask_logits = mask_output[:, :, -1]
        wait_logits = wait_output[:, :, -1]
        wait_alphas = torch.sigmoid(wait_logits)

        non_padding_mask = sampled_lengths_mask.type_as(mask_logits)
        cif_kd_loss = cif_kl_loss(mask_logits, wait_alphas, non_padding_mask)

        return cif_kd_loss

def token_level_mse_loss(student, teacher, non_padding_mask):
    # student, teacher: B x T x D
    # non_padding_mask: B x T
    teacher = teacher.detach()
    loss = F.mse_loss(student * non_padding_mask.unsqueeze(-1),
                      teacher * non_padding_mask.unsqueeze(-1),
                      reduction='none') 
    token_num = non_padding_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)
    return (loss / token_num).sum()

def cosine_similarity(x1, x2, dim=-1, eps=1e-08):
    axb = (x1 * x2).sum(dim) #torch.matmul(x1.unsqueeze(-2), x2.unsqueeze(-1))
    axa = torch.square(x1).sum(dim) # torch.matmul(x1.unsqueeze(-2), x1.unsqueeze(-1))
    bxb = torch.square(x2).sum(dim) #torch.matmul(x2.unsqueeze(-2), x2.unsqueeze(-1))
    cos = axb / torch.clamp(torch.sqrt(axa)*torch.sqrt(bxb), eps)
    return cos #cos.squeeze(-1).squeeze(-1)


def cosine_distance_loss(student, teacher, non_padding_mask, weight_flag=False):
    # student, teacher: B x T x D
    # non_padding_mask: B x T
    teacher = teacher.detach()
    """
    nx = F.normalize(student, dim=-1)
    ny = F.normalize(teacher, dim=-1)
    cos_dist = 1 - (nx * ny).sum(-1) # B x T
    """
    cos_dist = 1 - cosine_similarity(student, teacher, dim=-1)

    return (cos_dist * non_padding_mask).sum()

def cif_kl_loss(student_logits, teacher_probs, non_padding_mask):
    teacher_probs = teacher_probs.detach()
    loss = F.relu(student_logits) + torch.log1p(torch.exp(-torch.abs(student_logits))) - teacher_probs * student_logits
    return (loss * non_padding_mask).sum()


@register_model_architecture("fast", "fast")
def fast(args):
    args.simul_mode = getattr(args, "simul_mode", None)
    args.future_num = getattr(args, "future_num", 0)
    args.kd_loss = getattr(args, "kd_loss", "cos")
