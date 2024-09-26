#!/usr/bin/env python3
"""Score prediction file.
    - mAP
    - mAR
"""

import argparse
import json
import pandas as pd
import pickle
from metrics import eval_map, voc_ar
import numpy as np


def get_args():
    """Set up command-line interface and get arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions_file", type=str, required=True)
    parser.add_argument("-g", "--goldstandard_file", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="results.json")
    return parser.parse_args()


def score(gt_all, pred_all_map):
    """
    Calculate metrics for: AP at different distance threshold
    """
    score_dict = {}
    dist_thresh_list = []
    recall = {
        "Mesial": [],
        "Distal": [],
        "Cusp": [],
        "InnerPoint": [],
        "OuterPoint": [],
        "FacialPoint": []
    }
    for i in range(0, 30):
        dist_thresh = 0.1 * i
        rec, prec, ap = eval_map(pred_all_map, gt_all, dist_thresh=dist_thresh)
        score_dict[str(i)] = ap
        dist_thresh_list.append(dist_thresh)
        for cat in rec.keys():
            recall[cat].append(rec[cat][-1])
        # Collect all values for each class
    class_values = {'Mesial': [], 'Distal': [], 'Cusp': [], 'InnerPoint': [], 'OuterPoint': [], 'FacialPoint': []}
    for threshold in score_dict.values():
        for class_name in class_values.keys():
            class_values[class_name].append(threshold[class_name])

    # Calculate the mean for each class
    map = {class_name: sum(values) / len(values) for class_name, values in class_values.items()}
    mar = {}
    for cat in recall.keys():
        ar = voc_ar(np.exp(-np.asarray(dist_thresh_list)), recall, cat)
        mar[cat] = ar
    all_metrics = {"mAP": map, "mAR": mar}
    return all_metrics


def main():
    """Main function."""
    args = get_args()

    pred_submission = pd.read_csv(
        args.predictions_file
    )

    pred_all_map = {
        "Mesial": {},
        "Distal": {},
        "Cusp": {},
        "InnerPoint": {},
        "OuterPoint": {},
        "FacialPoint": {}
    }

    for _, row in pred_submission.iterrows():
        class_name = row['class']
        key = row['key']
        coord = [row['coord_x'], row['coord_y'], row['coord_z']]
        prob = row['score']

        if key not in pred_all_map[class_name]:
            pred_all_map[class_name][key] = [[coord, prob]]
        else:
            pred_all_map[class_name][key].append([coord, prob])

    with open(args.goldstandard_file, 'rb') as fp:
        gold = pickle.load(fp)

    scores = score(gold, pred_all_map)

    with open(args.output, "w") as out:
        res = {"submission_status": "SCORED", **scores}
        out.write(json.dumps(res))


if __name__ == "__main__":
    main()
