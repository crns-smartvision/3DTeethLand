# Landmark Detection Challenge

This repository contains scripts for validating and scoring prediction files for the 3DTeethLand-MICCAI 2024 Challenge.

## Scripts

### 1. `validate.py`

The `validate.py` script checks the format and content of a prediction file to ensure it adheres to the required specifications.

#### Usage

```bash
./validate.py -p <predictions_file> -e DockerRepository [-o <output>]
```

#### Arguments

- -p, --predictions_file (required): Path to the predictions CSV file.
- -e, --entity_type (required): The type of entity submitting the file. Only "DockerRepository" is valid.
- -o, --output: Path to the output JSON file (optional). If not provided, results will be printed to the console.


#### Validation Checks
The script performs the following validation checks on the predictions file:

1. **Duplicate Mixtures:** Checks for duplicate entries in the predictions file.
2. **NaN Values:** Checks for any NaN values in the coord_x, coord_y, and coord_z columns.
3. **Probability Values:** Ensures that the score column contains values between 0 and 1 inclusive.
4. **Class Values:** Validates that the class column contains only valid landmark types:
'Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint','FacialPoint'


#### Output
The script generates a JSON output indicating the validation status (VALIDATED or INVALID) and any errors found. Example output:

```json
{
    "submission_status": "VALIDATED",
    "submission_errors": ""
}
```
or
```json
{
  "submission_status": "INVALID",
  "submission_errors": "Found 2 duplicate mixture(s): ['id1', 'id2']\n'class' column contains invalid values: ['InvalidClass']\n..."
}
```

### 2. `score.py`

This script scores a prediction CSV file by calculating various metrics including mean Average Precision (mAP) and mean Average Recall (mAR)
at multiple distance threshold.


#### Usage

Run the script from the command line:
```bash
./score_predictions.py -p <predictions_file.csv> -g <goldstandard_file.pkl> [-o <output.json>]
```

#### Arguments
- -p, --predictions_file: Path to the predictions CSV file to be scored. (required)
- -g, --goldstandard_file: Path to the goldstandard pickle file. (required)
- -o, --output: Optional path to save the scoring results as a JSON file. Defaults to results.json.

#### Output
The script outputs a JSON object indicating the submission status ("SCORED") along with the calculated metrics.

If an output file is specified using the -o argument, the results will be saved to that file. Otherwise, the results will be printed to the console.

```json
{
  "submission_status": "SCORED",
  "mAP_0.50": 0.75,
  "mAP_1.00": 0.85,
  "mAP_2.00": 0.90,
  "mAP_3.00": 0.95
}
```
