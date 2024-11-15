import json

import numpy as np
from scipy.stats import wilcoxon
from glob import glob
from metrics import calculate_metrics_per_scan
import pandas as pd
import pickle
from tqdm import tqdm
import random
import seaborn as sns
import matplotlib.pyplot as plt


def compare_teams(metrics_dict, scan_names, alpha=0.001):
    teams = list(metrics_dict.keys())
    random.shuffle(teams)
    metrics = ['mAP', 'mAR']
    points = {team: 0 for team in teams}
    categories = [
        "Mesial",
        "Distal",
        "Cusp",
        "InnerPoint",
        "OuterPoint",
        "FacialPoint"
    ]
    # Perform pairwise comparison for each metric
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            for metric in metrics:
                for category in categories:
                    # Compare each pair of teams
                    team1 = teams[i]
                    team2 = teams[j]

                    # Prepare metric values for comparison
                    team1_values = np.round(
                        [metrics_dict[team1][scan_name][metric][category] for scan_name in scan_names],
                        decimals=4)
                    team2_values = np.round(
                        [metrics_dict[team2][scan_name][metric][category] for scan_name in scan_names],
                        decimals=4)
                    # Perform the Wilcoxon Signed Rank Test
                    stat = wilcoxon(team1_values, team2_values, alternative="two-sided", method="exact")
                    p_value = stat.pvalue


                    # Check if one team is statistically better
                    if p_value < alpha:
                        mean1 = np.mean(team1_values)
                        mean2 = np.mean(team2_values)
                        if mean1 > mean2:
                            points[team1] += 1
                            better_team = team1
                        else:
                            points[team2] += 1
                            better_team = team2

                        # plt.figure(figsize=(10, 6))
                        # sns.kdeplot(team1_values, label=f'{team1} - {metric} - {category}', shade=True, color='blue')
                        # sns.kdeplot(team2_values, label=f'{team2} - {metric} - {category}', shade=True, color='orange')
                        # plt.title(f'Distribution of Values for {metric} ({category})\nP-value={p_value} < {alpha} '
                        #           f'\nBetter team is {better_team}', fontsize=14)
                        # plt.xlabel('Metric Value')
                        # plt.ylabel('Density')
                        # plt.legend()
                        # plt.grid()
                        # plt.tight_layout()
                        # plt.show()
    return points


def normalize_points(bootstrap_points, num_teams, num_metrics, num_categories, n_bootstraps):
    # Total number of comparisons each team can participate in
    total_comparisons = (num_teams-1) * num_metrics * num_categories * n_bootstraps
    # Normalize points for each team
    normalized_scores = {team: round(points / total_comparisons, 4) for team, points in bootstrap_points.items()}

    return normalized_scores


def bootstrap_compare(metrics_dict, scan_names, alpha=0.001, n_bootstraps=100, resample_frac=0.9):
    teams = list(metrics_dict.keys())
    total_bootstrap_points = {team: 0 for team in teams}

    for _ in range(n_bootstraps):
        # Resample 90% of scan names
        bootstrap_sample = random.sample(scan_names, int(len(scan_names) * resample_frac))

        # Perform pairwise comparison on resampled data
        bootstrap_points = compare_teams(metrics_dict, bootstrap_sample, alpha)

        # Accumulate the points from this bootstrap sample
        for team, points in bootstrap_points.items():
            total_bootstrap_points[team] += points

    return total_bootstrap_points


if __name__ == "__main__":
    # Calculate metrics
    predictions_path = "./teams_predictions/final"
    with open('ground_truth_private_test.pkl', 'rb') as fp:
        gt_all = pickle.load(fp)
    teams_predictions_files_path = glob(predictions_path + "/*/predictions.csv")

    # metrics = {}
    # for team_pred_file in tqdm(teams_predictions_files_path):
    #     team_name = team_pred_file.split('/')[-2]
    #     pred_submission = pd.read_csv(team_pred_file)
    #     all_metrics = calculate_metrics_per_scan(pred_submission, gt_all)
    #     metrics[team_name] = all_metrics
    with open("metrics_dict.json", "r") as f:
        metrics = json.load(f)
    # Bootstrapping process with 100 iterations and 90% resampling
    final_points = bootstrap_compare(metrics, scan_names=list(gt_all['Cusp'].keys()), alpha=0.001, n_bootstraps=100,
                                     resample_frac=0.9)
    print(final_points)
    # Normalization
    num_teams = len(metrics)
    num_metrics = 2  # mAP, mAR
    num_categories = 6  # "Mesial", "Distal", "Cusp", etc.

    normalized_scores = normalize_points(final_points, num_teams, num_metrics, num_categories, n_bootstraps=100)
    # Sort by score in descending order
    ranked_scores = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

    # Print the results as a table
    print(f"{'Team':<15} {'Score':<10}")
    print("-" * 25)
    for team, score in ranked_scores:
        print(f"{team:<15} {score:<10.4f}")
