#!/usr/bin/env python3
"""Validate prediction file.

Prediction file should be a 6-col CSV file.
"""

import argparse
import json
import pandas as pd


COLS = {
    "key": str,
    "coord_x": float,
    "coord_y": float,
    "coord_z": float,
    "class": str,
    "score": float
}

LANDMARKS_TYPE = ["Mesial", "Distal", "Cusp", "InnerPoint", "OuterPoint", "FacialPoint"]

def get_args():
    """Set up command-line interface and get arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions_file", type=str, required=True)
    parser.add_argument("-e", "--entity_type", type=str, required=True)
    parser.add_argument("-o", "--output", type=str)
    return parser.parse_args()


def check_dups(pred):
    """Check for duplicate mixtures."""
    duplicates = pred.index.duplicated()
    if duplicates.any():
        return (
            f"Found {duplicates.sum()} duplicate mixture(s): "
            f"{pred[duplicates].index.to_list()}"
        )
    return ""


def check_nan_values(pred, col):
    """Check for NAN predictions."""
    missing_probs = pred[col].isna().sum()
    if missing_probs:
        return f"'{col}' column contains {missing_probs} NaN value(s)."
    return ""


def check_prob_values(pred, col):
    """Check that probabilities are between [0, 1]."""
    if (pred[col] < 0).any() or (pred[col] > 1).any():
        return f"'{col}' column should be between [0, 1] inclusive."
    return ""

def check_class_values(pred, col, valid_classes):
    """Check if class column contains only valid landmarks."""
    invalid_classes = pred[~pred[col].isin(valid_classes)]
    if not invalid_classes.empty:
        return (
            f"'{col}' column contains invalid values: "
            f"{invalid_classes[col].unique().tolist()}"
        )
    return ""


def validate(pred_file):
    """Validate predictions file against goldstandard."""
    errors = []

    try:
        pred = pd.read_csv(
            pred_file,
            usecols=COLS,
            dtype=COLS,
            float_precision="high",
            sep=','
        )
    except ValueError:
        errors.append(
            "Invalid prediction file headers and/or column types. "
            f"Expecting: {str(COLS)}."
        )
    else:
        errors.append(check_dups(pred))
        errors.append(check_class_values(pred, "class", LANDMARKS_TYPE))
        errors.append(check_nan_values(pred, "coord_x"))
        errors.append(check_nan_values(pred, "coord_y"))
        errors.append(check_nan_values(pred, "coord_z"))
        errors.append(check_prob_values(pred, "score"))
    return errors


def main():
    """Main function."""
    args = get_args()
    entity_type = args.entity_type.split(".")[-1]

    if entity_type != "DockerRepository":
        invalid_reasons = [f"Submission must be a File, not {entity_type}."]
    else:
        invalid_reasons = validate(
            pred_file=args.predictions_file
        )

    invalid_reasons = "\n".join(filter(None, invalid_reasons))
    status = "INVALID" if invalid_reasons else "VALIDATED"

    # truncate validation errors if >500 (character limit for sending email)
    if len(invalid_reasons) > 500:
        invalid_reasons = invalid_reasons[:496] + "..."
    res = json.dumps(
        {"submission_status": status, "submission_errors": invalid_reasons}
    )

    if args.output:
        with open(args.output, "w") as out:
            out.write(res)
    else:
        print(res)


if __name__ == "__main__":
    main()
