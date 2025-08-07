from pathlib import Path

import pandas as pd


def check_wave_overlap_and_covariates():
    """Check subject overlap across waves and analyze covariate changes over time.

    Loads processed MRI data from three waves (baseline, 2-year, 4-year follow-up)
    and analyzes:
    1. Subject overlap between waves
    2. Subjects appearing in follow-up but not baseline
    3. Categorical covariate stability across timepoints for multi-wave subjects

    Prints summary statistics and analysis results to console.
    """
    # Define paths
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

    processed_data_path = Path(
        analysis_root_path,
        "dev",
        "exp_1",
        "processed_data",
    )

    # Load each wave explicitly by name
    baseline_data = pd.read_csv(
        Path(
            processed_data_path,
            "mri_all_features_cov_baseline_year_1_arm_1.csv",
        ),
        index_col=0,
        low_memory=False,
    )

    wave2_data = pd.read_csv(
        Path(
            processed_data_path,
            "mri_all_features_cov_2_year_follow_up_y_arm_1.csv",
        ),
        index_col=0,
        low_memory=False,
    )

    wave3_data = pd.read_csv(
        Path(
            processed_data_path,
            "mri_all_features_cov_4_year_follow_up_y_arm_1.csv",
        ),
        index_col=0,
        low_memory=False,
    )

    # Get subjects from each wave
    baseline_subjects = set(baseline_data.index)
    wave2_subjects = set(wave2_data.index)
    wave3_subjects = set(wave3_data.index)

    print("=" * 50)
    print("SUBJECT OVERLAP ANALYSIS")
    print("=" * 50)

    print(f"Baseline subjects: {len(baseline_subjects)}")
    print(f"Wave 2 subjects: {len(wave2_subjects)}")
    print(f"Wave 3 subjects: {len(wave3_subjects)}")

    # Check follow-up subjects not in baseline
    wave2_not_baseline = wave2_subjects - baseline_subjects
    wave3_not_baseline = wave3_subjects - baseline_subjects

    print(f"Wave 2 subjects not in baseline: {len(wave2_not_baseline)}")
    print(f"Wave 3 subjects not in baseline: {len(wave3_not_baseline)}")

    # Create long-form data (baseline subjects only)
    wave2_filtered = wave2_data[wave2_data.index.isin(baseline_subjects)]
    wave3_filtered = wave3_data[wave3_data.index.isin(baseline_subjects)]

    long_data = pd.concat([baseline_data, wave2_filtered, wave3_filtered], axis=0)

    print(f"\nLong data shape: {long_data.shape}")
    print(f"Baseline (filtered): {len(baseline_data)}")
    print(f"Wave 2 (filtered to baseline): {len(wave2_filtered)}")
    print(f"Wave 3 (filtered to baseline): {len(wave3_filtered)}")

    print("\n" + "=" * 50)
    print("CATEGORICAL COVARIATE CHANGES")
    print("=" * 50)

    # Check categorical covariate stability across waves
    categorical_covs = ["demo_sex_v2", "site_id_l", "img_device_label", "rel_family_id"]

    # Get subjects present in multiple waves
    multi_wave_subjects = long_data.groupby(long_data.index)["eventname"].nunique() > 1
    multi_wave_indices = multi_wave_subjects[multi_wave_subjects].index

    print(f"Subjects in multiple waves: {len(multi_wave_indices)}")

    # Check changes for each categorical variable
    for cov in categorical_covs:
        if cov in long_data.columns:
            # Count subjects with changing values across waves
            changes = 0
            for subj in multi_wave_indices:
                subj_data = long_data.loc[subj, cov]
                if hasattr(subj_data, "nunique"):  # Multiple rows for subject
                    if subj_data.nunique() > 1:
                        changes += 1

            pct = changes / len(multi_wave_indices) * 100
            print(
                f"{cov}: {changes}/{len(multi_wave_indices)} subjects changed ({pct:.1f}%)"  # noqa: E501
            )
        else:
            print(f"{cov}: not found in data")


if __name__ == "__main__":
    check_wave_overlap_and_covariates()
