import torch
from collections import Counter
import numpy as np

epsilon = 1e-6


def segment_IOU(segment1, segment2):
    """Compute IOU between two segments"""

    inter_l = torch.max(segment1[0], segment2[0])  # [N,M,1]
    inter_r = torch.min(segment1[1], segment2[1])
    union_l = torch.min(segment1[0], segment2[0])
    union_r = torch.max(segment1[1], segment2[1])
    return (inter_r - inter_l) / (union_r - union_l + epsilon)


def mean_average_precision(outputs, targets, iou_threshold, num_classes):
    """
    Calculates mean average precision
    Parameters:
        outputs: dictionary from dataloader
        targets: Similar as outputs
        iou_threshold (float): threshold where predicted overlap is correct
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    outputs = [
        [
            (seq_start, pred_class, pred_box)
            for seq_start, pred_class, pred_box in zip(
                seq_starts, torch.argmax(pred_logits, axis=1), pred_boxes
            )
        ]
        for seq_starts, pred_logits, pred_boxes in zip(
            outputs["seq_start"], outputs["pred_logits"], outputs["pred_boundaries"]
        )
    ]

    targets = [
        [
            (seq_start, gt_class, gt_seqment)
            for seq_start, gt_class, gt_seqment in zip(
                target["seq_start"], target["classes"], target["coordinates"]
            )
        ]
        for target in targets
    ]
    for c in range(num_classes):
        detections = []
        ground_truths = []

        for output in outputs:
            for detection in output:
                if detection[1] == c:
                    detections.append(detection)
        for target in targets:
            for true_box in target:
                if true_box[1] == c:
                    ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = segment_IOU(
                    detection[2].clone().detach(),
                    gt[2].clone().detach(),
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


if __name__ == "__main__":
    batch_size, num_queries, num_classes = 2, 10, 3

    pred_logits = torch.rand(batch_size, num_queries, num_classes + 1)
    pred_boxes = torch.rand(batch_size, num_queries, 2)

    outputs = {
        "pred_logits": pred_logits,
        "pred_boundaries": pred_boxes,
        "seq_start": np.random.randint(low=0, high=2, size=(batch_size, num_queries)),
    }

    num_target_boxes = 8

    targets = [
        {
            "classes": torch.randint(low=0, high=num_classes, size=(num_target_boxes,)),
            "coordinates": torch.rand(num_target_boxes, 2),
            "seq_start": np.random.randint(low=0, high=2, size=(num_target_boxes,)),
        }
        for i in range(batch_size)
    ]
    print(mean_average_precision(outputs, targets, 0.5, 2))
