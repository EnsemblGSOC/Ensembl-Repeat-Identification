"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, cost_class: float = 1, cost_segments: float = 1, cost_siou: float = 1
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_segments: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_siou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_segments = cost_segments
        self.cost_siou = cost_siou
        assert (
            cost_class != 0 or cost_segments != 0 or cost_siou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["pred_logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        out_segments = outputs["pred_boxes"].flatten(
            0, 1
        )  # [batch_size * num_queries, 2]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_segments = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_segments = torch.cdist(out_segments, tgt_segments, p=1)

        # Compute the giou cost betwen boxes
        cost_siou = -seqment_IOU(out_segments, tgt_segments)

        # Final cost matrix
        C = (
            self.cost_segments * cost_segments
            + self.cost_class * cost_class
            + self.cost_siou * cost_siou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


def build_matcher():
    return HungarianMatcher()


def seqment_IOU(segment1: Tensor, segment2: Tensor):
    """Compute IOU between two set of segments

    Params:
        segment1: Tensor of dim [num1, 2]
        segment2: Tensor of dim [num2, 2]
    It result IOU value of dim [num1, num2]
    """
    area1 = segment1[:, 1] - segment1[:, 0]
    area2 = segment2[:, 1] - segment2[:, 0]

    lt = torch.max(segment1[:, None, :1], segment2[:, :1])  # [N,M,1]
    rb = torch.min(segment1[:, None, 1:], segment2[:, 1:])  # [N,M,1]

    wh = (rb - lt).clamp(min=0)  # [N,M,1]
    inter = wh[:, :, 0]
    union = area1[:, None] + area2 - inter
    return inter / union


if __name__ == "__main__":
    batch_size, num_queries, num_classes = 2, 10, 3

    pred_logits = torch.rand(batch_size, num_queries, num_classes)
    pred_boxes = torch.rand(batch_size, num_queries, 2)

    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

    num_target_boxes = 8

    targets = [
        {
            "labels": torch.randint(low=0, high=num_classes, size=(num_target_boxes,)),
            "boxes": torch.rand(num_target_boxes, 2),
        }
        for _ in range(batch_size)
    ]
    matcher = build_matcher()
    print(matcher(outputs, targets))
