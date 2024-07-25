# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import random

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from .cross_entropy_acc import LabelSmoothedCrossEntropyWithAccCriterion
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.utils import index_put

random.seed(0)

logger = logging.getLogger(__name__)


def logits2sent(preds, targets, dictionary, rate=0.03):
    if random.random() < rate:
        try:
            pred = dictionary.tokenizer.decode(preds)
            target = dictionary.tokenizer.decode(targets)
        except:
            pred = dictionary.string(preds)
            target = dictionary.string(targets)
        print('pred:\n{}\ntarget:\n{}\n'.format(pred, target))


@register_criterion("qua_ce_acc")
class QuantityCrossEntropyWithAccCriterion(LabelSmoothedCrossEntropyWithAccCriterion):
    def __init__(self, args, task):
        super().__init__(args, task, )
        self.args = args
        self.ignore_prefix_size = getattr(args, "ignore_prefix_size", 0)
        self.qua_by_src_txt_len = getattr(args, "qua_by_src_txt_len", False)
        self.lambda_qua = getattr(args, "lambda_qua", 0.05)

    @staticmethod
    def add_args(parser):
        super(
            QuantityCrossEntropyWithAccCriterion,
            QuantityCrossEntropyWithAccCriterion,
        ).add_args(parser)
        # fmt: off
        parser.add_argument(
            "--ignore-prefix-size",
            default=0,
            type=int,
            help="ignore first N tokens",
        )
        parser.add_argument(
            "--qua-by-src-txt-len",
            action="store_true",
            help="if True, use src text len to supervise qua loss",
        )
        parser.add_argument(
            "--lambda-qua",
            default=0.05,
            type=float,
            metavar="D",
            help="cif length loss weight",
        )

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)

    def compute_loss(self, model, net_output, sample, reduction, log_probs):
        # number loss
        _number = net_output["num_output"]
        if self.qua_by_src_txt_len:
            number = sample["net_input"]["target_lengths"].float()
        else:
            number = sample["target_lengths"].float()
        diff = torch.sqrt(torch.pow(_number - number, 2) + 1e-6).sum()
        qua_loss = diff


        target = sample["target"]  # no eos bos
        batch_size =target.size(0)
        if self.ignore_prefix_size > 0:
            target = target[:, self.ignore_prefix_size :].contiguous()
        # N, T -> N * T
        target = target.view(-1)
        lprobs = model.get_normalized_probs(net_output['logits'], log_probs=log_probs)

        if lprobs.size(0) == batch_size:
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
        else:
            lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()

        batch_first = getattr(lprobs, "batch_first", True)
        if not batch_first:
            lprobs = lprobs.transpose(0, 1)

        # N, T, D -> N * T, D
        lprobs = lprobs.view(-1, lprobs.size(-1))
        ce_loss, _ = label_smoothed_nll_loss(
            lprobs, target.long(), 0.1, ignore_index=self.padding_idx, reduce=reduction,
        )

        return lprobs, qua_loss, ce_loss

    def get_logging_output(self, sample, lprobs, loss, qua_loss, ce_loss):
        target = sample["target"]
        if self.ignore_prefix_size > 0:
            target = target[:, self.ignore_prefix_size :].contiguous()
        target = target.view(-1)
        mask = target != self.padding_idx
        
        assert lprobs.argmax(1).shape == target.shape
        correct = torch.sum(
            lprobs.argmax(1).masked_select(mask) == target.masked_select(mask)
        )
        total = torch.sum(mask)
        sample_size = sample["ntokens"]

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "qua_loss": utils.item(qua_loss.data),  # * sample['ntokens'],
            "ce_loss": utils.item(ce_loss.data),  # * sample['ntokens'],
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "correct": utils.item(correct.data),
            "total": utils.item(total.data),
        }

        return sample_size, logging_output

    def forward(self, model, sample, reduction="sum", log_probs=True):
        """Computes the cross entropy with accuracy metric for the given sample.

        This is similar to CrossEntropyCriterion in fairseq, but also
        computes accuracy metrics as part of logging

        Args:
            logprobs (Torch.tensor) of shape N, T, D i.e.
                batchsize, timesteps, dimensions
            targets (Torch.tensor) of shape N, T  i.e batchsize, timesteps

        Returns:
        tuple: With three elements:
            1) the loss
            2) the sample size, which is used as the denominator for the gradient
            3) logging outputs to display while training

        TODO:
            * Currently this Criterion will only work with LSTMEncoderModels or
            FairseqModels which have decoder, or Models which return TorchTensor
            as net_output.
            We need to make a change to support all FairseqEncoder models.
        """
        net_output = model(**sample["net_input"])
        num_output = net_output["num_output"].int()

        if model.training:
            lprobs, qua_loss, ce_loss = self.compute_loss(
                model, net_output, sample, reduction, log_probs
            )

            nsentences = sample["target"].size(0) + 1.0
            ntokens = sample["ntokens"]
            loss = self.args.lambda_qua * qua_loss * ntokens / nsentences + ce_loss

            sample_size, logging_output = self.get_logging_output(
                sample, lprobs, loss, qua_loss, ce_loss
            )
        else:
            lprobs, qua_loss, ce_loss = self.compute_loss(
                model, net_output, sample, reduction, log_probs
            )

            nsentences = sample["target"].size(0) + 1.0
            ntokens = sample["ntokens"]
            loss = self.args.lambda_qua * qua_loss * ntokens / nsentences + ce_loss

            sample_size, logging_output = self.get_logging_output(
                sample, lprobs, loss, qua_loss, ce_loss
            )
            import editdistance

            c_err = 0
            c_len = 0
            with torch.no_grad():
                for logits, l, t in zip(net_output['logits'][0], num_output, sample["target"]):
                    decoded = logits.argmax(dim=-1)[:l]
                    p = (t != self.task.target_dictionary.pad()) & (
                            t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units_arr = targ.tolist()
                    pred_units_arr = decoded.tolist()
                    # targ_units_arr = targ.unique_consecutive().tolist()
                    # pred_units_arr = decoded.unique_consecutive().tolist()
                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        correct_sum = sum(log.get("correct", 0) for log in logging_outputs)
        total_sum = sum(log.get("total", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ce_loss = sum(log.get("ce_loss", 0) for log in logging_outputs)
        qua_loss = sum(log.get("qua_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size if sample_size > 0 else 0.0,
            "ce_loss": ce_loss / sample_size if sample_size > 0 else 0.0,
            "qua_loss": qua_loss / nsentences if nsentences > 0 else 0.0,
            # if args.sentence_avg, then sample_size is nsentences, then loss
            # is per-sentence loss; else sample_size is ntokens, the loss
            # becomes per-output token loss
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "accuracy": correct_sum * 100.0 / total_sum if total_sum > 0 else 0.0,
            "correct": correct_sum,
            "total": total_sum,
            # total is the number of validate tokens
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = ce_loss / ntokens
        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        c_total = sum(log.get("c_total", 1) for log in logging_outputs)
        if c_total > 1:
            agg_output["uer"] = c_errors * 100.0 / c_total
        # loss: per output token loss
        # nll_loss: per sentence loss
        return agg_output


@register_criterion("fkd_ce_acc")
class FKDCEWithAccCriterion(QuantityCrossEntropyWithAccCriterion):
    def __init__(self, args, task):
        super().__init__(args, task) 
        self.lambda_w2v2 = getattr(args, "lambda_w2v2", 0.0)
        self.lambda_cif = getattr(args, "lambda_cif", 0.0)

    @staticmethod
    def add_args(parser):
        super(
            FKDCEWithAccCriterion,
            FKDCEWithAccCriterion,
        ).add_args(parser)
        parser.add_argument(
            "--lambda-w2v2",
            default=1.0,
            type=float,
            metavar="D",
            help="w2v2 kd loss weight",
        )
        parser.add_argument(
            "--lambda-cif",
            default=1.0,
            type=float,
            metavar="D",
            help="cif kl loss weight",
        )

    def get_logging_output(self, sample, sample_size, loss, w2v2_kd_loss, cif_kd_loss):

        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "w2v2_kd_loss": utils.item(w2v2_kd_loss.data),
            "cif_kd_loss": utils.item(cif_kd_loss.data),
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        return logging_output

    def forward(self, model, sample, reduction="sum", log_probs=True):
        """Computes the cross entropy with accuracy metric for the given sample.

        This is similar to CrossEntropyCriterion in fairseq, but also
        computes accuracy metrics as part of logging

        Args:
            logprobs (Torch.tensor) of shape N, T, D i.e.
                batchsize, timesteps, dimensions
            targets (Torch.tensor) of shape N, T  i.e batchsize, timesteps

        Returns:
        tuple: With three elements:
            1) the loss
            2) the sample size, which is used as the denominator for the gradient
            3) logging outputs to display while training

        TODO:
            * Currently this Criterion will only work with LSTMEncoderModels or
            FairseqModels which have decoder, or Models which return TorchTensor
            as net_output.
            We need to make a change to support all FairseqEncoder models.
        """
        net_output = model(**sample["net_input"])
        w2v2_kd_loss = net_output["w2v2_kd_loss"]
        cif_kd_loss = net_output["cif_kd_loss"]

        sample_size = sample["target"].size(0)
        
        loss = self.args.lambda_w2v2 * w2v2_kd_loss + self.args.lambda_cif * cif_kd_loss 

        logging_output = self.get_logging_output(
            sample, sample_size, loss, w2v2_kd_loss, cif_kd_loss
        )
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        w2v2_kd_loss = sum(log.get("w2v2_kd_loss", 0) for log in logging_outputs)
        cif_kd_loss = sum(log.get("cif_kd_loss", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / nsentences if nsentences > 0 else 0.0,
            "w2v2_kd_loss": w2v2_kd_loss / nsentences if nsentences > 0 else 0.0,
            "cif_kd_loss": cif_kd_loss / nsentences if nsentences > 0 else 0.0,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        return agg_output