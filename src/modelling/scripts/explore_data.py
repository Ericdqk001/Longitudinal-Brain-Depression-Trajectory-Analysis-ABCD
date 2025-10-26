"""Explore data to understand missing slope columns."""

import json
from pathlib import Path

import polars as pl


def explore_data(
    data_store_path: Path,
    version_name: str = "dev",
    experiment_number: int = 1,
):
    """Three-way comparison: features_of_interest vs slopes vs baseline data."""
    # Build paths
    analysis_root_path = Path(
        data_store_path,
        "users",
        "eric",
        "depression_trajectories",
    )
    version_path = Path(analysis_root_path, version_name)
    experiments_path = Path(version_path, "experiments", f"exp_{experiment_number}")
    processed_data_path = Path(experiments_path, "processed_data")
    results_path = Path(experiments_path, "results")

    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    # 1. Load features of interest
    features_path = Path(processed_data_path, "features_of_interest.json")
    print(f"Loading features from: {features_path}")
    with open(features_path) as f:
        features_of_interest = json.load(f)

    all_expected_features = []
    for modality, features in features_of_interest.items():
        all_expected_features.extend(features)
        print(f"  {modality}: {len(features)} features")

    print(f"Total expected features: {len(all_expected_features)}")

    # 2. Load slopes file
    slopes_path = Path(results_path, "subject_brain_slopes.csv")
    print(f"\nLoading slopes from: {slopes_path}")
    slopes_df = pl.read_csv(slopes_path)
    print(f"Slopes file shape: {slopes_df.shape}")

    slope_columns = [col for col in slopes_df.columns if col.endswith("_slope")]
    features_with_slopes = [col.replace("_slope", "") for col in slope_columns]
    print(f"Total slope columns: {len(slope_columns)}")

    # 3. Load baseline data
    data_path = Path(
        processed_data_path, "mri_all_features_cov_long_standardized_avg.csv"
    )
    print(f"\nLoading baseline data from: {data_path}")
    data = pl.read_csv(data_path)
    baseline_data = data.filter(pl.col("eventname") == "baseline_year_1_arm_1")
    print(f"Baseline data shape: {baseline_data.shape}")

    # STEP 2: THREE-WAY COMPARISON
    print("\n" + "=" * 70)
    print("STEP 2: THREE-WAY COMPARISON")
    print("=" * 70)

    # A. Features in JSON but missing from slopes
    missing_slopes = set(all_expected_features) - set(features_with_slopes)
    print(f"\n[A] Features in JSON missing slopes: {len(missing_slopes)}")

    # B. Check if missing slopes exist in baseline data
    baseline_cols = set(baseline_data.columns)
    missing_not_in_baseline = missing_slopes - baseline_cols
    missing_in_baseline = missing_slopes & baseline_cols

    print(f"\n[B] Of {len(missing_slopes)} missing slopes:")
    print(f"  - {len(missing_not_in_baseline)} NOT in baseline data (name mismatch)")
    print(f"  - {len(missing_in_baseline)} ARE in baseline data (model failed)")

    # C. Extra slopes not in features_of_interest
    extra_slopes = set(features_with_slopes) - set(all_expected_features)
    print(f"\n[C] Slopes not in features_of_interest: {len(extra_slopes)}")

    # STEP 3: DETAILED BREAKDOWN BY MODALITY
    print("\n" + "=" * 70)
    print("STEP 3: BREAKDOWN BY MODALITY")
    print("=" * 70)

    for modality, features in features_of_interest.items():
        expected = set(features)
        with_slopes = expected & set(features_with_slopes)
        missing = expected - set(features_with_slopes)

        success_rate = len(with_slopes) / len(expected) * 100 if expected else 0

        print(f"\n{modality}:")
        print(f"  Expected: {len(expected)}")
        print(f"  With slopes: {len(with_slopes)}")
        print(f"  Missing slopes: {len(missing)}")
        print(f"  Success rate: {success_rate:.1f}%")

        if missing:
            # Check if missing are in baseline
            missing_not_in_base = missing - baseline_cols
            missing_in_base = missing & baseline_cols

            if missing_not_in_base:
                print("  Missing (NOT in baseline - name mismatch):")
                for feat in sorted(missing_not_in_base)[:3]:
                    print(f"    - {feat}")
                if len(missing_not_in_base) > 3:
                    print(f"    ... and {len(missing_not_in_base) - 3} more")

            if missing_in_base:
                print("  Missing (IN baseline - model failed):")
                for feat in sorted(missing_in_base)[:3]:
                    print(f"    - {feat}")
                if len(missing_in_base) > 3:
                    print(f"    ... and {len(missing_in_base) - 3} more")

    # STEP 4: INVESTIGATE NAME MISMATCHES
    if missing_not_in_baseline:
        print("\n" + "=" * 70)
        print("STEP 4: INVESTIGATING NAME MISMATCHES")
        print("=" * 70)

        print("\nFeatures in JSON but NOT in baseline data:")
        for feat in sorted(missing_not_in_baseline)[:10]:
            print(f"  - {feat}")
        if len(missing_not_in_baseline) > 10:
            print(f"  ... and {len(missing_not_in_baseline) - 10} more")

        # Try to find similar column names
        print("\nSearching for similar column names in baseline...")
        for feat in sorted(missing_not_in_baseline)[:5]:
            # Look for columns containing part of the feature name
            feat_base = feat.replace("img_", "").split("_")[0]
            similar = [col for col in baseline_cols if feat_base in col]
            if similar:
                print(f"\n  '{feat}' might match:")
                for sim in similar[:3]:
                    print(f"    - {sim}")

    # FINAL SUMMARY
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Expected features: {len(all_expected_features)}")
    print(f"Features with slopes: {len(features_with_slopes)}")
    print(f"Missing slopes: {len(missing_slopes)}")
    success_rate = len(features_with_slopes) / len(all_expected_features) * 100
    print(f"Overall success rate: {success_rate:.1f}%")
    print("\nMissing slopes breakdown:")
    print(f"  - Name mismatch (not in baseline): {len(missing_not_in_baseline)}")
    print(f"  - Model failed (in baseline): {len(missing_in_baseline)}")


if __name__ == "__main__":
    data_store_path = Path("/Volumes/GenScotDepression")
    version_name = "dev"
    experiment_number = 1

    explore_data(
        data_store_path=data_store_path,
        version_name=version_name,
        experiment_number=experiment_number,
    )
