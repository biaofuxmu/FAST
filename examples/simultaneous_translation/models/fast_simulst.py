# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import logging
from fairseq import checkpoint_utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text import (
    fast_offline, 
    FASTOfflineModel,
    fast, 
    FASTModel
)

from torch import nn, Tensor
from typing import Dict, List

logger = logging.getLogger(__name__)

@register_model("fast_offline_simul")
class FASTOfflineSimulModel(FASTOfflineModel):
    """
    Implementation of the paper:

    SimulMT to SimulST: Adapting Simultaneous Text Translation to
    End-to-End Simultaneous Speech Translation

    https://www.aclweb.org/anthology/2020.aacl-main.58.pdf
    """

    @staticmethod
    def add_args(parser):
        super(FASTOfflineSimulModel, FASTOfflineSimulModel).add_args(parser)
        parser.add_argument(
            "--train-monotonic-only",
            action="store_true",
            default=False,
            help="Only train monotonic attention",
        )

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        tgt_dict = task.tgt_dict

        from examples.simultaneous_translation.models.transformer_monotonic_attention import (
            TransformerMonotonicDecoder,
        )

        decoder = TransformerMonotonicDecoder(args, tgt_dict, embed_tokens)

        pretraining_path = getattr(args, "load_pretrained_decoder_from", None)
        if pretraining_path is not None:
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from, strict=False
            )
            logger.info(f"loaded pretrained decoder from: {args.load_pretrained_decoder_from}")

        return decoder


@register_model("fast_simul")
class FASTSimulModel(FASTModel):
    """
    Implementation of the paper:

    SimulMT to SimulST: Adapting Simultaneous Text Translation to
    End-to-End Simultaneous Speech Translation

    https://www.aclweb.org/anthology/2020.aacl-main.58.pdf
    """

    @staticmethod
    def add_args(parser):
        super(FASTSimulModel, FASTSimulModel).add_args(parser)
        parser.add_argument(
            "--train-monotonic-only",
            action="store_true",
            default=False,
            help="Only train monotonic attention",
        )

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        tgt_dict = task.tgt_dict

        from examples.simultaneous_translation.models.transformer_monotonic_attention import (
            TransformerMonotonicDecoder,
        )

        decoder = TransformerMonotonicDecoder(args, tgt_dict, embed_tokens)

        pretraining_path = getattr(args, "load_pretrained_decoder_from", None)
        if pretraining_path is not None:
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from, strict=False
            )
            logger.info(f"loaded pretrained decoder from: {args.load_pretrained_decoder_from}")

        return decoder


@register_model_architecture("fast_offline_simul", "fast_offline_simul")
def fast_offline_simul(args):
    fast_offline(args)


@register_model_architecture("fast_simul", "fast_simul")
def fast_simul(args):
    fast(args)