import json
import logging
from pathlib import Path

import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Arguments as variables (matching mixed_effect.py structure)
data_store_path = Path(
    "/",
    "Volumes",
    "GenScotDepression",
)

analysis_root_path = Path(
    data_store_path,
    "users",
    "Eric",
    "depression_trajectories",
)


version_name = "dev"
experiment_number = 1


version_path = Path(
    analysis_root_path,
    version_name,
)

experiments_path = Path(
    version_path,
    "experiments",
    f"exp_{experiment_number}",
)

processed_data_path = Path(
    experiments_path,
    "processed_data",
)

results_path = Path(
    experiments_path,
    "results",
)

# Load brain slopes data (output from mixed_effect.py)
slopes_path = Path(
    results_path,
    "subject_brain_slopes.csv",
)

logging.info("Loading slopes from: %s", slopes_path)
slopes_df = pl.read_csv(slopes_path, has_header=True)

logging.info("Slopes data shape: %s", slopes_df.shape)
logging.info("Number of subjects: %d", slopes_df.select("src_subject_id").n_unique())

# Load the long-format imaging data
data_path = Path(
    processed_data_path,
    "mri_all_features_cov_long_standardized_avg.csv",
)

logging.info("Loading data from: %s", data_path)
data = pl.read_csv(data_path, has_header=True)

logging.info("Data shape: %s", data.shape)
logging.info("Number of subjects: %d", data.select("src_subject_id").n_unique())

# Filter to baseline data only
baseline_data = data.filter(pl.col("eventname") == "baseline_year_1_arm_1")

logging.info("Baseline data shape: %s", baseline_data.shape)
logging.info(
    "Number of baseline subjects: %d", baseline_data.select("src_subject_id").n_unique()
)

# Join slopes with baseline data
merged_df = slopes_df.join(baseline_data, on="src_subject_id", how="inner")

logging.info("Merged data shape: %s", merged_df.shape)
logging.info(
    "Number of merged subjects: %d", merged_df.select("src_subject_id").n_unique()
)

# Check if class_label column exists (trajectory classes)
if "class_label" in merged_df.columns:
    logging.info("Trajectory classes found in data")
    logging.info("Class distribution:")
    print(merged_df.group_by("class_label").len())
else:
    logging.warning("class_label column not found in merged data")

# Load features of interest from JSON file
features_of_interest_path = Path(processed_data_path, "features_of_interest.json")

logging.info("Loading features of interest from: %s", features_of_interest_path)

with open(features_of_interest_path, "r") as f:
    features_of_interest = json.load(f)

logging.info("Features loaded for modalities:")
for modality, features in features_of_interest.items():
    logging.info("Modality %s: %d features", modality, len(features))
