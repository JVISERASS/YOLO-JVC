"""
Script to write some necessary functions not related with the rest of the scripts.
"""

import os
from typing import Literal

import torch
from torch import nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.constants import EPSILON, TARGET_SIZE_IMG, B, C, S


def iou(
    boxes_preds: torch.Tensor,
    boxes_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates Intersection over Union for each pair of the batch.

    Args:
        boxes_preds: Predictions of bounding boxes, Dimensions: [batch_size, S, S, 4].
        boxes_labels: Correct bounding boxes.  Dimensions: [batch_size, S, S, 4].

    Returns:
        Intersection over Union for each pair of the batch. Dimensions:
        [batch_size, S, S].
    """

    # TODO

    # Prediction Coordeinates
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

    # Label Coordinates
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    # Intersection Coordinates
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Intersection Area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    # Union Area
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection + EPSILON
    # IoU
    iou = intersection / union

    return iou



def nms(
    predicted_boxes: list[torch.Tensor],
    threshold_confidence: float = 0.5,
    threshold_repeated: float = 0.5,
) -> list[torch.Tensor]:
    """
    Applies Non Max Suppression given the predicted boxes. When updating the list, if
    two boxes have different classes, we do not delete them even if their IoU is high.

    Parameters:
        predicted_boxes: List with the bounding boxes predicted. They have to be
            relative to the hole image, not to each cell. Each box has dimension C + 5.
        threshold_confidence: Threshold to remove predicted bounding boxes with low
            confidence.
        threshold_repeated: Threshold to remove predicted boxes because we consider they
            refer to the same box.

    Returns:
        Predicted bounding boxes after NMS. Each box has dimension C + 5.
    """

    # TODO
    if len(predicted_boxes) == 0:
        return []
    
    boxes = torch.stack(predicted_boxes) # Convert list to tensor for easier manipulation

    # Filter by objectness threshold
    obj_mask = boxes[:, C] >= threshold_confidence # Keep boxes with objectness >= threshold, dimensions [num_boxes]
    boxes = boxes[obj_mask] # Apply mask to boxes, dimensions [num_filtered_boxes, C + 5]

    if len(boxes) == 0:
        return []
    
    # Sort boxes by confidence score in descending order
    boxes = boxes[torch.argsort(boxes[:, C], descending=True)]
    keep_boxes = []
    while boxes.size(0):
        # Select the box with the highest confidence
        current_box = boxes[0]
        keep_boxes.append(current_box)
        if boxes.size(0) == 1:
            break
        # Compute IoU of the current box with the rest
        ious = iou(
            current_box[C + 1 :].unsqueeze(0).unsqueeze(0).unsqueeze(0), # Shape [1, 1, 1, 4]
            boxes[1:, C + 1 :].unsqueeze(1).unsqueeze(1), # Shape [num_boxes - 1, 1, 4]
        ).squeeze() # Shape [num_boxes - 1]
        # Create a mask to filter out boxes with high IoU and same class
        if C == 1:
            class_mask = torch.ones(boxes[1:].size(0), dtype=torch.bool)  # Asumir misma clase por defecto
            current_class_prob = current_box[0]
            other_class_probs = boxes[1:, 0] 
            class_mask = torch.abs(current_class_prob - other_class_probs) < 0.5
        else:
            class_mask = boxes[1:, :C].argmax(dim=1) == current_box[:C].argmax(dim=0)
            


        iou_mask = ious < threshold_repeated # Shape [num_boxes - 1]
        combined_mask = ~(class_mask & ~iou_mask) # Keep boxes with different class or low IoU
        boxes = boxes[1:][combined_mask] # Apply mask to boxes
    return keep_boxes

        


def to_image_coords(boxes: torch.Tensor, img_w: int, img_h: int) -> list[torch.Tensor]:
    """
    Transforms the coordinates of the bounding boxes from relative to each cell to
    relative to the hole image.

    Args:
        boxes: Predicted bounding boxes. Dimensions: [S, S, 5 * B + C].
        img_w: Width of the image.
        img_h: Height of the image.

    Returns:
        List with the scaled bounding boxes to the hole image. Each tensor has dimension
        C + 5.
    """

    # TODO
    cell_w = img_w / S
    cell_h = img_h / S

    i_grid, j_grid = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')
    i_grid = i_grid.float()
    j_grid = j_grid.float()
    # We create a grid with the coordinates of the top-left corner of each cell
    # Dimensions: [S, S]

    boxes_image = []

    for b in range(B): 
        start_idx = b *5
        end_idx = start_idx + 5 + C
        box_data = boxes[..., start_idx:end_idx] # Shape [S, S, 5 + C]

        obj_mask = box_data[..., C] >= 0 # Objectness mask, shape [S, S]
        if obj_mask.sum() == 0:
            continue

        rel_x = box_data[..., C+1] # Relative x, shape [S, S]
        rel_y = box_data[..., C+2] # Relative y, shape [S, S]
        w = box_data[..., C+3] # Relative width, shape [S, S]
        h = box_data[..., C+4] # Relative height, shape [S, S]

        abs_x = (j_grid + rel_x) * cell_w # Absolute x, shape [S, S]
        abs_y = (i_grid + rel_y) * cell_h # Absolute y, shape [S, S]
        abs_w = w * cell_w # Absolute width, shape [S, S]
        abs_h = h * cell_h # Absolute height, shape [S, S]

        valid_indices = torch.nonzero(obj_mask, as_tuple=False) # Indices of valid boxes, shape [num_valid_boxes, 2]

        for idx in valid_indices:
            i,j = idx[0].item(), idx[1].item()

            new_box = torch.zeros(C + 5)
            new_box[:C] = box_data[i, j, :C] # Class probabilities
            new_box[C] = box_data[i, j, C] # Objectness
            new_box[C + 1] = abs_x[i, j] # x
            new_box[C + 2] = abs_y[i, j] # y
            new_box[C + 3] = abs_w[i, j] # width
            new_box[C + 4] = abs_h[i, j] # height
            boxes_image.append(new_box)


    return boxes_image


# This last part is just to calculate the mAP. It is a little bit annoying.


def mean_average_precision(subset: Literal["train", "test"], model: nn.Module) -> float:
    """
    Calculates mAP in the test set.

    Args:
        subset: Subset where the mAP is calculated.
        model: YOLO.

    Returns:
        Calculated mAP.
    """

    targets = _get_targets_map(subset)
    predictions = _get_predictions_map(subset, model)

    metric = MeanAveragePrecision(
        iou_type="bbox", iou_thresholds=[0.5], class_metrics=True
    )
    metric.update(predictions, targets)
    result = metric.compute()

    return float(result["map"])


def _get_targets_map(subset: Literal["train", "test"]) -> list[dict[str, torch.Tensor]]:
    """
    Gets the targets in the correct format for mAP calculation.

    Args:
        subset: Subset where the mAP is calculated.

    Returns:
        List of dictionaries with the form: {"boxes": ..., "labels": ...}.
    """

    targets = []

    for i in range(len(os.listdir(f"data/labels_resized/{subset}"))):
        temp_box = []
        temp_label = []
        with open(f"data/labels_resized/{subset}/{i}.txt", encoding="utf8") as f:
            for label in f.readlines():
                label, c_x, c_y, width, height = [  # type: ignore
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                temp_label.append(label)
                # Change coordinates to (x1, y1, x2, y2)
                temp_box.append(
                    torch.tensor(
                        [
                            (c_x - width / 2) * TARGET_SIZE_IMG[0],
                            (c_y - height / 2) * TARGET_SIZE_IMG[1],
                            (c_x + width / 2) * TARGET_SIZE_IMG[0],
                            (c_y + height / 2) * TARGET_SIZE_IMG[1],
                        ]
                    )
                )
        boxes = torch.stack(temp_box)
        labels = torch.tensor(temp_label)
        targets.append({"boxes": boxes, "labels": labels})

    return targets


def _get_predictions_map(
    subset: Literal["train", "test"], model: nn.Module
) -> list[dict[str, torch.Tensor]]:
    """
    Gets the predictions in the correct format for mAP calculation.

    Args:
        subset: Subset where the mAP is calculated.
        model: YOLO model.

    Returns:
        List of dictionaries with the form: {"boxes": ..., "scores": ...,
        "labels": ...}.
    """

    dir_path = f"data/images/{subset}"
    predictions = []

    for i in range(len(os.listdir(dir_path))):
        bounding_boxes = model.predict(f"{dir_path}/raccoon-{i}.jpg")  # type: ignore
        boxes = []
        scores = []
        labels = []
        if len(bounding_boxes) > 0:
            for bounding_box in bounding_boxes:
                c_x, c_y, width, height = bounding_box[C + 1 :]
                # Change coordinates to (x1, y1, x2, y2)
                normalized_bb = torch.tensor(
                    [
                        float(c_x - width / 2),
                        float(c_y - height / 2),
                        float(c_x + width / 2),
                        float(c_y + height / 2),
                    ]
                )
                boxes.append(normalized_bb)
                scores.append(torch.min(torch.tensor(1.0), bounding_box[C]))
                labels.append(torch.argmax(bounding_box[:C]))
            predictions.append(
                {
                    "boxes": torch.stack(boxes),
                    "scores": torch.stack(scores),
                    "labels": torch.stack(labels),
                }
            )
        else:  # no predictions were made
            predictions.append(
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
            )

    return predictions
