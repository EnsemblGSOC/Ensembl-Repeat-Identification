# standard library
import math
from typing import List
import time

# third party
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import pytorch_lightning as pl

# project
from matcher_segment import build_matcher, segment_IOU
from transformer import Transformer
from mAP_validation import mean_average_precision


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        len = d_model // 2

        cos_pos = torch.cos(position * div_term)
        cos_pos = cos_pos[:, :len]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = cos_pos
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        output = self.pe.repeat(batch_size, 1, 1)
        return output[:, :seq_len, :]


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and boundaries)
    """

    def __init__(self, num_classes, matcher, eos_coef, losses, weight_dict):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.weight_dict = weight_dict
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "classes" containing a tensor of dim [nb_target_segments]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["classes"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_segments(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the segments_IoU loss
        targets dicts must contain the key "coordinates" containing a tensor of dim [nb_target_segments, 2]
        """
        assert "pred_boundaries" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boundaries"][idx]
        target_boxes = torch.cat(
            [t["coordinates"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_ssegments = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_ssegments"] = loss_ssegments.sum() / num_boxes
        IOU = segment_IOU(src_boxes, target_boxes)
        loss_IOU = 1 - IOU
        losses["loss_IOU"] = loss_IOU.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "classes": self.loss_labels,
            "coordinates": self.loss_segments,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["classes"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )

        num_boxes = torch.clamp(num_boxes / 1, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses


def build_criterion(configuration):
    matcher = build_matcher(configuration)
    losses = ["classes", "coordinates"]
    weight_dict = {
        "loss_ce": configuration.cost_class,
        "loss_ssegments": configuration.cost_segments,
        "loss_IOU": configuration.cost_siou,
    }

    criterion = SetCriterion(
        configuration.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=configuration.eos_coef,
        losses=losses,
    )
    return criterion


class DETR(pl.LightningModule):
    """
    This is the main model that performs segments detection and classification

    Copy-paste from DETR module with modifications
    """

    def __init__(
        self,
        transformer,
        num_classes,
        num_queries,
        num_nucleobase_letters,
        criterion,
        configuration,
    ):
        """Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                        DETR can detect in an subsequence.
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.emb = nn.Embedding(num_nucleobase_letters, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.segment_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim)
        self.criterion = criterion
        self.configuration = configuration

    def forward(self, sample: Tensor, seq_starts: List[int]):
        """
        Parameters:
            -- sample: batched sequences, of shape [batch_size x seq_len x embedding_len ]
                 eg, when choosing one hot encoding, embedding_len will be 5, [A, T, C, G, N]
        Return values
            -- pred_logits: the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
            -- pred_boundaries: The normalized boundaries coordinates for all queries, represented as
                            (center, width). These values are normalized in [0, 1].
        """
        sample = self.emb(sample)
        pos = self.pe(sample)
        hs = self.transformer(sample, self.query_embed.weight, pos)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.segment_embed(hs).sigmoid()

        # ([batch_size]) -> [batch_size, num_queries]

        seq_starts = [
            [seq_start for _ in range(self.num_queries)] for seq_start in seq_starts
        ]
        out = {
            "pred_logits": outputs_class,
            "pred_boundaries": outputs_coord,
            "seq_start": seq_starts,
        }
        return out

    def training_step(self, batch, batch_idx):
        samples, seq_starts, targets = batch
        outputs = self.forward(samples, seq_starts)
        mAP = mean_average_precision(
            outputs=outputs,
            targets=targets,
            iou_threshold=0.5,
            num_classes=self.num_classes,
        )
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        train_losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        self.log("train_loss", train_losses, batch_size=self.configuration.batch_size)
        self.log("mAP", mAP, batch_size=self.configuration.batch_size)

        return train_losses

    def validation_step(self, batch, batch_idx):
        samples, seq_starts, targets = batch
        outputs = self.forward(samples, seq_starts)
        mAP = mean_average_precision(
            outputs=outputs,
            targets=targets,
            iou_threshold=0.5,
            num_classes=self.num_classes,
        )

        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        val_losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        self.log("val_losses", val_losses, batch_size=self.configuration.batch_size)
        self.log("mAP", mAP, batch_size=self.configuration.batch_size)

        return val_losses

    def on_test_start(self):
        self.sample_sequence = torch.empty(0).to(self.device)
        self.sample_labels = []
        self.sample_predictions = []

    def test_step(self, batch, batch_idx):
        samples, seq_starts, targets = batch
        outputs = self.forward(samples, seq_starts)
        mAP = mean_average_precision(
            outputs=outputs,
            targets=targets,
            iou_threshold=0.5,
            num_classes=self.num_classes,
        )

        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        test_losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        self.log("test_loss", test_losses, batch_size=self.configuration.batch_size)
        self.log("mAP", mAP, batch_size=self.configuration.batch_size)

        self.sample_sequence = torch.cat((self.sample_sequence, samples))
        self.sample_labels.append(targets)
        self.sample_predictions.append(outputs)

    def on_test_end(self):
        if self.num_sample_predictions > 0:
            with torch.random.fork_rng():
                torch.manual_seed(int(time.time() * 1000))
                permutation = torch.randperm(len(self.sample_sequence))
        self.sample_sequence = self.sample_sequence[
            permutation[0 : self.num_sample_predictions]
        ].tolist()

        sequences = [
            self.configuration.dna_sequence_mapper.label_encoding_to_sequence(seq)
            for seq in self.sample_sequence
        ]
        print("\nsample assignments")
        for seq in sequences:
            print(seq)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.configuration.lr)
        return optimizer


def test_criterion():
    batch_size, num_queries, num_classes = 2, 10, 3

    pred_logits = torch.rand(batch_size, num_queries, num_classes + 1)
    pred_boxes = torch.rand(batch_size, num_queries, 2)

    outputs = {"pred_logits": pred_logits, "pred_boundaries": pred_boxes}

    num_target_boxes = 8

    targets = [
        {
            "classes": torch.randint(low=0, high=num_classes, size=(num_target_boxes,)),
            "coordinates": torch.rand(num_target_boxes, 2),
        }
        for _ in range(batch_size)
    ]
    matcher = build_matcher()
    losses = ["classes", "coordinates"]
    a_dict = {"loss_ce": 1, "loss_ssegments": 1, "loss_IOU": 1}

    matcher = build_matcher()
    criterion = SetCriterion(
        num_classes, matcher=matcher, eos_coef=1, losses=losses, weight_dict=a_dict
    )
    res = criterion(outputs, targets)
    print(torch.argmax(outputs["pred_logits"], axis=2).shape)


def build_model(configuration):
    num_classes = configuration.num_classes
    num_queries = configuration.num_queries
    # hardcode, can be warped later.

    transformer = Transformer(
        d_model=configuration.embedding_dimension,
        nhead=configuration.nhead,
        dropout=configuration.dropout,
    )
    model = DETR(
        transformer,
        num_classes=num_classes,
        num_queries=num_queries,
        num_nucleobase_letters=configuration.num_nucleobase_letters,
    )

    return model, criterion


if __name__ == "__main__":

    n, s, e = 10, 100, 5
    num_queries = 100
    fake_configuration = 1
    transformer = Transformer(d_model=5, nhead=5)
    criterion = 1
    model = DETR(
        transformer=transformer,
        num_classes=11,
        num_queries=num_queries,
        configuration=fake_configuration,
        criterion=criterion,
        num_nucleobase_letters=6,
    )
    x = torch.rand(n, s, e)
    output = model(x)
    print(output)
