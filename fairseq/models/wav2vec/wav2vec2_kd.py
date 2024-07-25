# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask, lengths_to_mask
from fairseq.models import BaseFairseqModel, register_model, register_model_architecture
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange, index_put
from fairseq.dataclass import ChoiceEnum
from fairseq.models.wav2vec import Wav2Vec2Model, TransformerEncoder, base_architecture
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


@register_model("wav2vec2_kd")
class Wav2Vec2KDModel(Wav2Vec2Model):
    def __init__(self, args):
        super().__init__(args)

        self.future_mask_length = args.future_mask_length
        self.stu_encoder = TransformerEncoder(args)
        print("future_mask_length:", self.future_mask_length)
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(Wav2Vec2KDModel, Wav2Vec2KDModel).add_args(parser)

    @classmethod
    def build_model(cls, args, task=None):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        return cls(args)

    def forward(self, source, padding_mask=None, mask=True, features_only=False, finish_read=True):

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        if self.simul_mode is not None: # streaming inference
            w2v2_encoder = self.stu_encoder

            if finish_read or self.simul_mode.lower() == "vanilla":
                x = w2v2_encoder(x, padding_mask=padding_mask)
            else:
                future_len = self.args.future_num
                padding_mask_len = (~padding_mask).sum(dim=-1)
                if self.simul_mode.lower() == "fai":
                    masked_emb = self.mask_emb.repeat(x.size(0), future_len, 1).to(x.device)
                    max_src_lengths = max(padding_mask_len.max().item() + future_len, x.size(1))
                    new_x = x.new_zeros((x.size(0), max_src_lengths, x.size(2)))
                    new_padding_mask = padding_mask.new_ones((x.size(0), max_src_lengths))
                    new_x[:, :x.size(1), :] = x
                    new_padding_mask[:, :x.size(1)] = padding_mask
                    mask_index = torch.arange(future_len).unsqueeze(0).to(x.device) + padding_mask_len.unsqueeze(-1)
                    x = new_x.scatter_(1, mask_index.unsqueeze(-1).repeat(1,1, x.size(2)), masked_emb)
                    padding_mask = new_padding_mask.scatter_(1, mask_index, False)
                    x = w2v2_encoder(x, padding_mask=padding_mask)[:, :padding_mask_len.max().item(), :]
                    token_len = x.size(1)
                    padding_mask = torch.arange(token_len).unsqueeze(0).to(x.device) >= padding_mask_len.unsqueeze(-1)
                else:
                    raise ValueError("not defined inference policy: %s." % self.simul_mode)

        else:
            mask_output, wait_output, sampled_lengths_mask, mask_mask = self.get_output_for_sampled_wait_and_mask(
                full_speech_tokens=x.clone(), padding_mask=padding_mask)    
            
            return {"x": x,
                    "mask_output": mask_output, 
                    "wait_output": wait_output, 
                    "padding_mask": padding_mask, 
                    "sampled_lengths_mask": sampled_lengths_mask,
                    "mask_mask": mask_mask,
                    }

        if features_only:
            return {"x": x, "padding_mask": padding_mask}

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands, *_ = self.quantizer(unmasked_features, produce_targets=False)
                negs, _ = self.sample_negatives(neg_cands, y.size(1))
                negs = self.project_q(negs)

            else:
                negs, _ = self.sample_negatives(y, y.size(1))

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(unmasked_features, y.size(1))
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(y, y.size(1))

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {"x": x, "padding_mask": padding_mask, "features_pen": features_pen}

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def get_output_for_sampled_wait_and_mask(self, full_speech_tokens, padding_mask):
        device = full_speech_tokens.device
        future_mask_length = self.future_mask_length 
        
        # sample lengths
        if padding_mask is not None:
            speech_tokens_lengths = (1 - padding_mask.int()).sum(dim=1) # full speech length T
            sampled_lengths = torch.tensor([torch.randint(1, toks_len + 1, (1,)) for toks_len in speech_tokens_lengths]).to(device)

        else:
            batch_size, speech_tokens_lengths, _ = full_speech_tokens.size()
            sampled_lengths = torch.randint(1, speech_token_lengths + 1, (batch_size,)).to(device)
        
        # actual future_mask_length: length of masking tokens
        num_of_mask = torch.minimum(speech_tokens_lengths - sampled_lengths, torch.tensor(future_mask_length).to(device))

        # length of wait speech tokens
        final_lengths = sampled_lengths + num_of_mask 
        # final_lengths = speech_tokens_lengths

        # calculate the wait and mask speech tokens
        max_sampled_lengths = max(sampled_lengths)
        max_final_lengths = max(final_lengths)

        wait_speech_tokens = full_speech_tokens[:, :max_final_lengths]

        # length mask of streaming, will also be used for token level KD loss
        sampled_lengths_mask = F.pad(lengths_to_mask(sampled_lengths), 
                                     (0, max_final_lengths - max_sampled_lengths), 
                                     value=False)
        final_lengths_mask = lengths_to_mask(final_lengths)
        mask_mask = final_lengths_mask ^ sampled_lengths_mask
        mask_speech_tokens = index_put(wait_speech_tokens.clone(), mask_mask, self.mask_emb)
        # mask_speech_tokens = wait_speech_tokens.clone()

        mask_outputs = self.stu_encoder(mask_speech_tokens, padding_mask=~final_lengths_mask)
        mask_output = mask_outputs


        wait_outputs = self.encoder(wait_speech_tokens, padding_mask=~final_lengths_mask)
        wait_output = wait_outputs

        return mask_output, wait_output, sampled_lengths_mask, mask_mask
