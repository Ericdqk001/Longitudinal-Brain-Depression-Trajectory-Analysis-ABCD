import json
import logging
from pathlib import Path

import polars as pl
from pymer4.models import lmer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Parameters
version_name = "dev"
experiment_number = 1

# Define paths
data_store_path = Path(
    "/",
    "Volumes",
    "GenScotDepression",
)

if data_store_path.exists():
    logging.info("Mounted data store path: %s", data_store_path)

analysis_root_path = Path(
    data_store_path,
    "users",
    "Eric",
    "depression_trajectories",
)

version_path = Path(
    analysis_root_path,
    version_name,
)

experiments_path = Path(
    version_path,
    f"exp_{experiment_number}",
)

processed_data_path = Path(
    experiments_path,
    "processed_data",
)

# Load the averaged long-format data
data_path = Path(
    processed_data_path,
    "mri_all_features_cov_long_standardized_avg.csv",
)

logging.info("Loading data from: %s", data_path)
data = pl.read_csv(data_path, has_header=True)

logging.info("Data shape: %s", data.shape)
logging.info("Number of subjects: %d", data.select("src_subject_id").n_unique())

# Load features of interest from JSON file
features_of_interest_path = Path(processed_data_path, "features_of_interest.json")

logging.info("Loading features of interest from: %s", features_of_interest_path)

with open(features_of_interest_path, "r") as f:
    features_of_interest = json.load(f)

logging.info("Features loaded for modalities:")
for modality, features in features_of_interest.items():
    logging.info("Modality %s: %d features", modality, len(features))

# Convert categorical variables
categorical_vars = [
    "src_subject_id",
    "demo_sex_v2",
    "img_device_label",
    "site_id_l",
    "rel_family_id",
    "demo_comb_income_v2",
    "eventname",
]

data = data.with_columns(
    [
        pl.col(col).cast(pl.String).cast(pl.Categorical)
        for col in categorical_vars
        if col in data.columns
    ]
)

logging.info("Converted categorical variables to category dtype")

# Create baseline values for each brain feature as covariates
logging.info("Creating baseline covariates for slope-only models...")

# Get baseline data
baseline_data = data.filter(pl.col("eventname") == "baseline_year_1_arm_1")

# Get all brain features
all_brain_features = []
for features in features_of_interest.values():
    all_brain_features.extend(features)

logging.info(
    "Found %d total brain features across all modalities", len(all_brain_features)
)

# Create baseline covariate mapping for each brain feature
for feature in all_brain_features:
    if feature in baseline_data.columns:
        baseline_feature_name = f"{feature}_baseline"

        # Create baseline values for this feature
        baseline_values = baseline_data.select(["src_subject_id", feature]).rename(
            {feature: baseline_feature_name}
        )

        # Join baseline values to main data
        data = data.join(baseline_values, on="src_subject_id", how="left")

logging.info("Created baseline covariates for all brain features")

# Base fixed effects (common to all modalities)
base_fixed_effects = [
    "time",
    "BMI_zscore",
    "demo_sex_v2",
    "demo_comb_income_v2",
    "interview_age",
    "age2",
    "img_device_label",
]

# Random effects - slope only (no random intercepts for subjects)
random_effects = "(0 + time|src_subject_id) + (1|site_id_l/rel_family_id)"

# Storage for results
failed_models = []
slopes_list = []  # Store polars DataFrames for each feature

# Fit models for each modality and feature
logging.info("Starting mixed-effects modeling...")

total_features = sum(len(features) for features in features_of_interest.values())
feature_count = 0

for modality, brain_features in features_of_interest.items():
    logging.info("Processing modality: %s (%d features)", modality, len(brain_features))

    # Add modality-specific global measures
    fixed_effects = base_fixed_effects.copy()

    if modality == "cortical_thickness":
        fixed_effects.append("smri_thick_cdk_mean")
    elif modality == "cortical_surface_area":
        fixed_effects.append("smri_area_cdk_total")
    elif modality == "cortical_volume":
        fixed_effects.append("smri_vol_scs_intracranialv")
    elif modality == "subcortical_volume":
        fixed_effects.append("smri_vol_scs_intracranialv")
    elif modality == "tract_FA":
        fixed_effects.append("FA_all_dti_atlas_tract_fibers")
    elif modality == "tract_MD":
        fixed_effects.append("MD_all_dti_atlas_tract_fibers")

    logging.info("Fixed effects for %s: %s", modality, fixed_effects)

    for feature in brain_features:
        feature_count += 1

        if feature_count % 50 == 0:
            logging.info(
                "Progress: %d/%d features processed", feature_count, total_features
            )

        try:
            # Add baseline covariate for this specific feature
            baseline_feature_name = f"{feature}_baseline"
            feature_fixed_effects = fixed_effects + [baseline_feature_name]

            # Construct formula
            formula = (
                f"{feature} ~ {' + '.join(feature_fixed_effects)} + {random_effects}"
            )

            logging.info(
                "Fitting model for feature: %s (modality: %s)", feature, modality
            )
            logging.info("Formula: %s", formula)

            # Fit model
            model = lmer(formula, data=data)
            model.fit(summarize=False, verbose=False)

            # Check convergence
            converged = model.fitted

            if converged:
                logging.info(
                    "Model converged for feature: %s (modality: %s)", feature, modality
                )

                # Extract subject slopes (this returns a polars DataFrame from pymer4)
                subject_slopes_df = model.ranef["src_subject_id"]

                # Convert to polars DataFrame
                slopes_pl = pl.DataFrame(
                    {
                        "src_subject_id": subject_slopes_df["level"].to_list(),
                        f"{feature}_slope": subject_slopes_df["time"].to_list(),
                        "modality": modality,
                    }
                )

                # Store in list for later joining
                slopes_list.append(slopes_pl)

            else:
                logging.warning(
                    "Model failed to converge for feature: %s (modality: %s)",
                    feature,
                    modality,
                )
                failed_models.append(f"{modality}:{feature}")

        except Exception as e:
            logging.error(
                "Model fitting failed for feature %s (modality %s): %s",
                feature,
                modality,
                str(e),
            )
            failed_models.append(f"{modality}:{feature}")
            # Error already logged above

# Summary statistics
n_total = total_features
n_failed = len(failed_models)
n_converged = n_total - n_failed

logging.info("=" * 50)
logging.info("MIXED-EFFECTS MODELING SUMMARY")
logging.info("=" * 50)
logging.info("Total brain features: %d", n_total)
logging.info(
    "Successfully converged: %d (%.1f%%)", n_converged, 100 * n_converged / n_total
)
logging.info("Failed to converge: %d (%.1f%%)", n_failed, 100 * n_failed / n_total)
if failed_models:
    logging.warning("Failed models: %s", failed_models[:10])
    if len(failed_models) > 10:
        logging.warning("... and %d more failed models", len(failed_models) - 10)

# Combine all slopes into single DataFrame and save
if slopes_list:
    # Start with first DataFrame
    final_slopes_df = slopes_list[0]

    # Join all other DataFrames on src_subject_id
    for slopes_df in slopes_list[1:]:
        final_slopes_df = final_slopes_df.join(
            slopes_df, on="src_subject_id", how="outer"
        )

    # Create results directory if it doesn't exist
    results_path = Path(experiments_path, "results")
    results_path.mkdir(parents=True, exist_ok=True)

    # Save slopes using polars
    slopes_output_path = Path(results_path, "subject_brain_slopes.csv")
    final_slopes_df.write_csv(slopes_output_path)

    logging.info("Subject slopes saved to: %s", slopes_output_path)
    logging.info("Slopes DataFrame shape: %s", final_slopes_df.shape)
    logging.info("Features with slopes: %d", len(slopes_list))

    # Show sample of results
    logging.info("Sample of slopes DataFrame:")
    print(final_slopes_df.head())

else:
    logging.warning("No slopes to save - all models failed")
