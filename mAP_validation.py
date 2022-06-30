import torch
from collections import Counter


def segment_IOU(segment1, segment2):
    """Compute IOU between two segments"""

    inter_l = torch.max(segment1[0], segment2[0])  # [N,M,1]
    inter_r = torch.min(segment1[1], segment2[1])
    union_l = torch.min(segment1[0], segment2[0])
    union_r = torch.max(segment1[1], segment2[1])
    return (inter_r - inter_l) / (union_r - union_l)


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

    # list storing all AP for respective classes
    average_precisions = []
    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(int(num_classes)):

        pred_classes, pred_boxes = (
            torch.argmax(outputs["pred_logits"], axis=2),
            outputs["pred_boundaries"],
        )
        num_detections = 0
        for pred_class in pred_classes:
            for p_cls in pred_class:
                if p_cls == c:
                    num_detections += 1

        num_target = 0
        for target in targets:
            for tar_cls in target["classes"]:
                if tar_cls == c:
                    num_target += 1

        TP = torch.zeros(num_detections)
        FP = torch.zeros(num_detections)
        total_true_bboxes = num_target

        for pred_classes, pred_boxes, target in zip(pred_classes, pred_boxes, targets):
            target_assigned = torch.zeros(target["coordinates"].shape[0])
            pred_index = 0
            for pred_class, pred_boundary in zip(pred_classes, pred_boxes):
                if pred_class != c:
                    continue
                pred_index += 1
                best_IOU = 0.0
                best_target_index = 0
                target_index = 0
                for tg_class, tg_boundary in zip(
                    target["classes"], target["coordinates"]
                ):
                    if tg_class != c:
                        continue
                    target_index += 1
                    curr_IOU = segment_IOU(tg_boundary, pred_boundary)
                    if curr_IOU > best_IOU:
                        best_IOU = curr_IOU
                        best_target_index = target_index

                if best_IOU > iou_threshold:
                    # only detect ground truth detection once
                    if target_assigned[best_target_index] == 0:
                        # true positive and add this bounding box to seen
                        TP[pred_index] = 1
                        target_assigned[best_target_index] == 1
                    else:
                        FP[pred_index] = 1

                # if IOU is lower then the detection is a false positive
                else:
                    FP[pred_index] = 1
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    if len(average_precisions) != 0:
        return sum(average_precisions) / len(average_precisions)
    else:
        return 0


if __name__ == "__main__":
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
    print(mean_average_precision(outputs, targets))
