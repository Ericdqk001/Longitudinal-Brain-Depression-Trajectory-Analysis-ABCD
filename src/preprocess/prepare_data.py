import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# TODO: Check for outliers of the features (+- 5SD)


def preprocess(
    wave: str = "baseline_year_1_arm_1",
    version_name: str = "test",
    experiment_number: int = 1,
):
    logging.info("-----------------------")
    logging.info("Processing wave: %s", wave)
    # %%

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

    analysis_data_path = Path(
        analysis_root_path,
        "data",
    )

    experiments_path = Path(
        analysis_root_path,
        version_name,
        f"exp_{experiment_number}",
    )

    processed_data_path = Path(
        experiments_path,
        "processed_data",
    )

    if not processed_data_path.exists():
        processed_data_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    core_data_path = Path(
        data_store_path,
        "data",
        "abcd",
        "release5.1",
        "core",
    )

    imaging_path = Path(
        core_data_path,
        "imaging",
    )

    general_info_path = Path(
        core_data_path,
        "abcd-general",
    )

    # For biological sex (demo_sex_v2)
    demographics_path = Path(
        general_info_path,
        "abcd_p_demo.csv",
    )

    demographics = pd.read_csv(
        demographics_path,
        index_col=0,
        low_memory=False,
    )

    # Select the baseline year 1 demographics data
    demographics_bl = demographics[demographics.eventname == "baseline_year_1_arm_1"]

    demographics_bl.demo_sex_v2.value_counts()

    inter_sex_subs = demographics_bl[demographics_bl.demo_sex_v2 == 3].index

    # Recommended image inclusion (NDA 4.0 abcd_imgincl01)
    mri_y_qc_incl_path = Path(
        imaging_path,
        "mri_y_qc_incl.csv",
    )

    mri_y_qc_incl = pd.read_csv(
        mri_y_qc_incl_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_qc_incl = mri_y_qc_incl[mri_y_qc_incl.eventname == wave]

    logging.info(
        "Sample size with MRI recommended inclusion %d", mri_y_qc_incl.shape[0]
    )

    # Remove subjects with intersex from the imaging data
    mri_y_qc_incl = mri_y_qc_incl[~mri_y_qc_incl.index.isin(inter_sex_subs)]

    logging.info(
        "Remove intersex subjects from the imaging data, number = %d",
        len(inter_sex_subs),
    )

    # %%
    ### Remove imaging data with data quality issues, overall MRI clinical report is
    # used here as well.

    # First, we apply quality control to T1 weighted images (for structural features).
    # Conditions for inclusion:
    # 1. T1w data recommended for inclusion (YES)
    # 2. dmri data recommended for inclusion (YES)
    # 3. Overall MRI clinical report score < 3, which excludes subjects with
    # neurological issues.

    logging.info("Quality Control Criteria:")
    logging.info("T1 data recommended for inclusion = 1")
    logging.info("dMRI data recommended for inclusion = 1")
    logging.info("Overall MRI clinical report score < 3")

    mri_clin_report_path = Path(
        imaging_path,
        "mri_y_qc_clfind.csv",
    )

    mri_clin_report = pd.read_csv(
        mri_clin_report_path,
        index_col=0,
        low_memory=False,
    )

    mri_clin_report_bl = mri_clin_report[mri_clin_report.eventname == wave]

    qc_passed_indices = list(
        mri_y_qc_incl[
            (mri_y_qc_incl.imgincl_t1w_include == 1)
            & (mri_y_qc_incl.imgincl_dmri_include == 1)
        ].index
    )

    qc_passed_mask = mri_clin_report_bl.index.isin(qc_passed_indices)

    logging.info(
        "Sample size after QC passed, number = %d",
        mri_clin_report_bl[qc_passed_mask].shape[0],
    )

    score_mask = mri_clin_report_bl.mrif_score < 3

    subs_pass = mri_clin_report_bl[qc_passed_mask & score_mask]

    logging.info(
        "sample size after QC passed and clinical report (score < 3), number = %d",
        subs_pass.shape[0],
    )

    ###

    # %%
    ### Now prepare the imaging data

    mri_y_smr_thk_dst_path = Path(
        imaging_path,
        "mri_y_smr_thk_dsk.csv",
    )

    mri_y_smr_thk_dst = pd.read_csv(
        mri_y_smr_thk_dst_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_smr_vol_dst_path = Path(
        imaging_path,
        "mri_y_smr_vol_dsk.csv",
    )

    mri_y_smr_vol_dst = pd.read_csv(
        mri_y_smr_vol_dst_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_smr_area_dst_path = Path(
        imaging_path,
        "mri_y_smr_area_dsk.csv",
    )

    mri_y_smr_area_dst = pd.read_csv(
        mri_y_smr_area_dst_path,
        index_col=0,
        low_memory=False,
    )

    mir_y_smr_vol_aseg_path = Path(
        imaging_path,
        "mri_y_smr_vol_aseg.csv",
    )

    mri_y_smr_vol_aseg = pd.read_csv(
        mir_y_smr_vol_aseg_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_dti_fa_fs_at_path = Path(
        imaging_path,
        "mri_y_dti_fa_fs_at.csv",
    )

    mri_y_dti_fa_fs_at = pd.read_csv(
        mri_y_dti_fa_fs_at_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_dti_md_fs_at_path = Path(
        imaging_path,
        "mri_y_dti_md_fs_at.csv",
    )

    mri_y_dti_md_fs_at = pd.read_csv(
        mri_y_dti_md_fs_at_path,
        index_col=0,
        low_memory=False,
    )

    # Select the data for the subjects who passed the quality control and drop
    # subjects with missing data

    # Cortical thickness data
    mri_y_smr_thk_dst = mri_y_smr_thk_dst[mri_y_smr_thk_dst.eventname == wave]

    logging.info(
        "Sample size with T1w cortical thickness data, number = %d",
        mri_y_smr_thk_dst.shape[0],
    )

    t1w_cortical_thickness_pass = mri_y_smr_thk_dst[
        mri_y_smr_thk_dst.index.isin(subs_pass.index)
    ].dropna()

    logging.info(
        "Sample size with complete CT data after QC, number = %d",
        t1w_cortical_thickness_pass.shape[0],
    )

    # Cortical volume data
    mri_y_smr_vol_dst = mri_y_smr_vol_dst[mri_y_smr_vol_dst.eventname == wave]

    logging.info(
        "Sample size with T1w cortical volume data, number = %d",
        mri_y_smr_vol_dst.shape[0],
    )

    t1w_cortical_volume_pass = mri_y_smr_vol_dst[
        mri_y_smr_vol_dst.index.isin(subs_pass.index)
    ].dropna()

    logging.info(
        "Sample size with complete CV data after QC, number = %d",
        t1w_cortical_volume_pass.shape[0],
    )

    # Cortical surface area data

    mri_y_smr_area_dst = mri_y_smr_area_dst[mri_y_smr_area_dst.eventname == wave]

    logging.info(
        "Sample size with T1w cortical surface area data, number = %d",
        mri_y_smr_area_dst.shape[0],
    )

    t1w_cortical_surface_area_pass = mri_y_smr_area_dst[
        mri_y_smr_area_dst.index.isin(subs_pass.index)
    ].dropna()

    logging.info(
        "Sample size with complete SA data after QC, number = %d",
        t1w_cortical_surface_area_pass.shape[0],
    )

    # Subcortical volume

    t1w_subcortical_volume = mri_y_smr_vol_aseg[mri_y_smr_vol_aseg.eventname == wave]

    logging.info(
        "Sample size with T1w subcortical volume data, number = %d",
        t1w_subcortical_volume.shape[0],
    )

    t1w_subcortical_volume_pass = t1w_subcortical_volume[
        t1w_subcortical_volume.index.isin(subs_pass.index)
    ]

    # NOTE: These columns were dropped because they had all missing values or all zeros

    subcortical_all_zeros_cols = [
        "smri_vol_scs_lesionlh",
        "smri_vol_scs_lesionrh",
        "smri_vol_scs_wmhintlh",
        "smri_vol_scs_wmhintrh",
    ]

    t1w_subcortical_volume_pass = t1w_subcortical_volume_pass.drop(
        columns=subcortical_all_zeros_cols
    ).dropna()

    logging.info("Subcortical all zeros columns dropped")
    logging.info("Column names: %s", subcortical_all_zeros_cols)

    logging.info(
        "Sample size with complete subcortical volume data after QC, number = %d",
        t1w_subcortical_volume_pass.shape[0],
    )

    # # Add tracts data (mri_y_dti_fa_fs_at (FA), mri_y_dti_md_fs_at(MD))

    dmir_fractional_anisotropy = mri_y_dti_fa_fs_at[
        mri_y_dti_fa_fs_at.eventname == wave
    ]

    logging.info(
        "Sample size with dMRI fractional anisotropy data, number = %d",
        dmir_fractional_anisotropy.shape[0],
    )

    # Dropna later because some columns will be removed
    dmir_fractional_anisotropy_pass = dmir_fractional_anisotropy[
        dmir_fractional_anisotropy.index.isin(subs_pass.index)
    ]

    logging.info(
        "Sample size with complete FA data after QC, number = %d",
        dmir_fractional_anisotropy_pass.shape[0],
    )

    dmir_mean_diffusivity = mri_y_dti_md_fs_at[mri_y_dti_md_fs_at.eventname == wave]

    logging.info(
        "Sample size with dMRI mean diffusivity data, number = %d",
        dmir_mean_diffusivity.shape[0],
    )

    dmir_mean_diffusivity_pass = dmir_mean_diffusivity[
        dmir_mean_diffusivity.index.isin(subs_pass.index)
    ]

    logging.info(
        "Sample size with complete MD data after QC, number = %d",
        dmir_mean_diffusivity_pass.shape[0],
    )

    # Rename the FA and DM features to have "lh" or "rh" suffixes

    dmri_data_dict_path = Path(
        analysis_data_path,
        "dmri_data_dict.txt",
    )

    with open(dmri_data_dict_path, "r") as f:
        dmri_data_dict = f.read()

    def parse_dti_features_pretty(raw_text: str) -> dict:
        mapping = {}
        for line in raw_text.strip().split("\n"):
            parts = re.split(r"\t+", line.strip())
            if len(parts) < 4:
                continue

            original_feature = parts[0]
            description = parts[3].strip()

            # Determine modality: FA or MD
            if "fractional anisotropy" in description.lower():
                prefix = "FA"
            elif "mean diffusivity" in description.lower():
                prefix = "MD"
            else:
                continue  # skip if neither FA nor MD

            # Determine hemisphere
            if "right" in description.lower():
                suffix = "rh"
            elif "left" in description.lower():
                suffix = "lh"
            else:
                suffix = ""

            # Clean region name
            region = description.lower()
            region = re.sub(
                r"(average|mean)\s+(fractional anisotropy|diffusivity)\s+within",
                "",
                region,
            )
            region = region.strip().strip(".,")
            region = region.replace(",", "")
            region = re.sub(
                r"\bright\b|\bleft\b", "", region
            )  # remove 'right' or 'left'
            region = re.sub(
                r"[-/]", "_", region
            )  # replace dashes and slashes with underscores
            region = re.sub(r"\s+", "_", region)  # replace spaces with underscores
            region = region.strip("_")  # remove leading/trailing underscores if any

            cleaned_name = f"{prefix}_{region}{suffix}"
            mapping[original_feature] = cleaned_name

        return mapping

    # Sort the columns of the DTI dataframes because they can cause issues with later
    # concatenation to create long form data
    def sort_dmri_columns(df):
        # Separate 'eventname' from the rest
        columns = list(df.columns)
        first_col = [columns[0]]  # keep 'eventname' first
        rest_cols = columns[1:]

        # Sort by the number at the end of each column name
        sorted_cols = sorted(
            rest_cols,
            key=lambda x: int(re.search(r"_(\d+)$", x).group(1))
            if re.search(r"_(\d+)$", x)
            else float("inf"),
        )

        # Reorder the DataFrame
        return df[first_col + sorted_cols]

    dmir_fractional_anisotropy_pass = sort_dmri_columns(dmir_fractional_anisotropy_pass)

    dmir_mean_diffusivity_pass = sort_dmri_columns(dmir_mean_diffusivity_pass)

    logging.info("Sort dMRI FA/MD columns by number at the end of each column name")
    logging.info("For example: 'dmdtifp1_43', ''dmdtifp1_44', 'dmdtifp1_45'")
    logging.info("Sorting is error-free, checked")

    # Parse the DTI features
    dti_features_mapping = parse_dti_features_pretty(dmri_data_dict)

    logging.info(
        "Parsing FA/MD feature descriptions to new feature names is error-free, Checked"
    )

    # Rename the columns in dmri data
    dmir_fractional_anisotropy_pass.rename(
        columns=dti_features_mapping,
        inplace=True,
    )

    logging.info("Renaming FA features to new feature names is error-free, Checked")

    # Drop these columns because they are duplicates with a slightly different regional
    # focus
    FA_cols_to_drop = [
        "FA_dti_atlas_tract_fornix_excluding_fimbrialh",
        "FA_dti_atlas_tract_fornix_excluding_fimbriarh",
        "FA_dti_atlas_tract_superior_corticostriate_frontal_cortex_onlylh",
        "FA_dti_atlas_tract_superior_corticostriate_frontal_cortex_onlyrh",
        "FA_dti_atlas_tract_superior_corticostriate_parietal_cortex_onlylh",
        "FA_dti_atlas_tract_superior_corticostriate_parietal_cortex_onlyrh",
    ]

    logging.info(
        "Drop the following FA columns because they are duplicates with a slightly different regional focus:"  # noqa: E501
    )
    logging.info("%s", FA_cols_to_drop)

    logging.info(
        "FA number of features before dropping columns: %d",
        dmir_fractional_anisotropy_pass.shape[1],
    )

    dmir_fractional_anisotropy_pass = dmir_fractional_anisotropy_pass.drop(
        columns=FA_cols_to_drop
    )

    logging.info(
        "FA number of features after dropping columns: %d",
        dmir_fractional_anisotropy_pass.shape[1],
    )

    dmir_fractional_anisotropy_pass = dmir_fractional_anisotropy_pass.dropna()

    logging.info(
        "Sample size with complete FA data after QC, number = %d",
        dmir_fractional_anisotropy_pass.shape[0],
    )

    dmir_mean_diffusivity_pass.rename(
        columns=dti_features_mapping,
        inplace=True,
    )

    logging.info("Renaming MD features to new feature names is error-free, Checked")

    MD_cols_to_drop = [
        "MD_dti_atlas_tract_fornix_excluding_fimbrialh",
        "MD_dti_atlas_tract_fornix_excluding_fimbriarh",
        "MD_dti_atlas_tract_superior_corticostriate_frontal_cortex_onlylh",
        "MD_dti_atlas_tract_superior_corticostriate_frontal_cortex_onlyrh",
        "MD_dti_atlas_tract_superior_corticostriate_parietal_cortex_onlylh",
        "MD_dti_atlas_tract_superior_corticostriate_parietal_cortex_onlyrh",
    ]
    logging.info(
        "Drop the following MD columns because they are duplicates with a slightly different regional focus:"  # noqa: E501
    )
    logging.info("%s", MD_cols_to_drop)

    logging.info(
        "MD number of features before dropping columns: %d",
        dmir_mean_diffusivity_pass.shape[1],
    )

    dmir_mean_diffusivity_pass = dmir_mean_diffusivity_pass.drop(
        columns=MD_cols_to_drop
    )

    logging.info(
        "MD number of features after dropping columns: %d",
        dmir_mean_diffusivity_pass.shape[1],
    )

    dmir_mean_diffusivity_pass = dmir_mean_diffusivity_pass.dropna()

    logging.info(
        "Sample size with complete MD data after QC, number = %d",
        dmir_mean_diffusivity_pass.shape[0],
    )

    # Combine all the modalities

    mri_all_features = pd.concat(
        [
            t1w_cortical_thickness_pass,
            t1w_cortical_volume_pass,
            t1w_cortical_surface_area_pass,
            t1w_subcortical_volume_pass,
            dmir_fractional_anisotropy_pass,
            dmir_mean_diffusivity_pass,
        ],
        axis=1,
    )

    logging.info(
        "Sample size with all imaging features, number = %d", mri_all_features.shape[0]
    )

    # %%

    # Drop eventname column
    mri_all_features = mri_all_features.drop(columns="eventname")

    ### Add covariates to be considered in the analysis

    logging.info("Adding covariates to the imaging features")

    # For site information (imaging device ID)
    mri_y_adm_info_path = Path(
        imaging_path,
        "mri_y_adm_info.csv",
    )

    mri_y_adm_info = pd.read_csv(
        mri_y_adm_info_path,
        index_col=0,
        low_memory=False,
    )

    mri_y_adm_info = mri_y_adm_info[mri_y_adm_info.eventname == wave]

    le = LabelEncoder()

    # Using .fit_transform function to fit label
    # encoder and return encoded label
    label = le.fit_transform(mri_y_adm_info["mri_info_deviceserialnumber"])

    logging.info("Add covariate: mri_info_deviceserialnumber")

    mri_y_adm_info["img_device_label"] = label

    logging.info(
        "Using LabelEncoder to encode the imaging device ID is error-free, Checked"
    )

    # For imaging site
    abcd_y_it_path = Path(
        general_info_path,
        "abcd_y_lt.csv",
    )
    abcd_y_it = pd.read_csv(
        abcd_y_it_path,
        index_col=0,
        low_memory=False,
    )

    abcd_y_it = abcd_y_it[abcd_y_it.eventname == wave]

    site_id = abcd_y_it["site_id_l"]

    # Strip the "site" prefix from the site ID
    site_id = site_id.str.replace("site", "", regex=False)
    site_id = site_id.astype(int)

    # For interview_age (in months)
    abcd_y_lt_path = Path(
        general_info_path,
        "abcd_y_lt.csv",
    )

    abcd_y_lt = pd.read_csv(
        abcd_y_lt_path,
        index_col=0,
        low_memory=False,
    )

    abcd_y_lt = abcd_y_lt[abcd_y_lt.eventname == wave]

    # Add an age squared term

    abcd_y_lt["age2"] = abcd_y_lt.interview_age**2

    logging.info("Add covariate: interview_age and age2 (sqaured interview_age)")

    # Add family ID

    genetics_path = Path(
        core_data_path,
        "genetics",
    )

    genetics_relatedness_path = Path(
        genetics_path,
        "gen_y_pihat.csv",
    )

    genetics_relatedness = pd.read_csv(
        genetics_relatedness_path,
        index_col=0,
        low_memory=False,
    )

    family_id = genetics_relatedness["rel_family_id"]

    # Add household income

    household_income = demographics_bl["demo_comb_income_v2"].copy()

    # Not available category (777:refused to answer, 999: don't know, missing values)

    household_income = household_income.replace(
        [777, 999],
        np.nan,
    )

    logging.info(
        "Subjects who either refused to answer or don't know their income are set to NA"
    )

    # Height and weight data for BMI calculation

    physical_health_path = Path(
        core_data_path,
        "physical-health",
        "ph_y_anthro.csv",
    )

    physical_health = pd.read_csv(
        physical_health_path,
        index_col=0,
        low_memory=False,
    )

    physical_health = physical_health[physical_health.eventname == wave]

    # Select the relevant columns for height and weight
    weight = physical_health["anthroweightcalc"]
    height = physical_health["anthroheightcalc"]

    # Drop those with zeros for height and weight

    # Log subject IDs with zero weight
    zero_weight_subjects = weight[weight == 0].index.tolist()
    if zero_weight_subjects:
        logging.info(
            "Subjects with zero weight (n=%d): %s",
            len(zero_weight_subjects),
            zero_weight_subjects,
        )

    # Log subject IDs with zero height
    zero_height_subjects = height[height == 0].index.tolist()
    if zero_height_subjects:
        logging.info(
            "Subjects with zero height (n=%d): %s",
            len(zero_height_subjects),
            zero_height_subjects,
        )

    weight = weight[weight != 0]
    height = height[height != 0]

    series_list = [
        demographics_bl.demo_sex_v2,
        mri_y_adm_info.img_device_label,
        abcd_y_lt.interview_age,
        abcd_y_lt.age2,
        household_income,
        family_id,
        site_id,
        weight,
        height,
    ]

    covariates = pd.concat(series_list, axis=1).dropna()

    boys_std_path = Path(
        analysis_data_path,
        "bmi-boys-z-who-2007-exp.xlsx",
    )

    girls_std_path = Path(
        analysis_data_path,
        "bmi-girls-z-who-2007-exp.xlsx",
    )

    boys_lms = pd.read_excel(boys_std_path)
    girls_lms = pd.read_excel(girls_std_path)

    # Calculate BMI z-score referencing to WHO growth standards

    def convert_units(weight_lbs, height_inches):
        weight_kg = weight_lbs * 0.453592  # Convert pounds to kg
        height_m = height_inches * 0.0254  # Convert inches to meters
        return weight_kg, height_m

    def calculate_bmi(weight_kg, height_m):
        return weight_kg / (height_m**2)

    def get_lms_values(age_months, sex):
        """Retrieves L, M, S values for a given age (months) and sex.
        Sex: 1 = Male (boys_lms), 2 = Female (girls_lms).
        """
        lms_table = boys_lms if sex == 1 else girls_lms
        row = lms_table[lms_table["Month"] == age_months]

        if row.empty:
            raise ValueError(
                f"LMS values not found for age {age_months} months and sex {sex}"
            )

        return row["L"].values[0], row["M"].values[0], row["S"].values[0]

    def calculate_bmi_zscore(bmi, age_months, sex):
        L, M, S = get_lms_values(age_months, sex)

        if L == 0:
            z = np.log(bmi / M) / S  # Special case when L = 0
        else:
            z = ((bmi / M) ** L - 1) / (L * S)

        return z

    # Convert weight and height to metric units
    covariates[["weight_kg", "height_m"]] = covariates.apply(
        lambda row: convert_units(row["anthroweightcalc"], row["anthroheightcalc"]),
        axis=1,
        result_type="expand",
    )

    # Calculate BMI
    covariates["BMI"] = covariates.apply(
        lambda row: calculate_bmi(row["weight_kg"], row["height_m"]), axis=1
    )

    # Calculate BMI z-scores
    covariates["BMI_zscore"] = covariates.apply(
        lambda row: calculate_bmi_zscore(
            row["BMI"],
            row["interview_age"],
            row["demo_sex_v2"],
        ),
        axis=1,
    )

    # Drop the temporary columns used for BMI calculation
    covariates = covariates.drop(
        columns=["weight_kg", "height_m", "anthroweightcalc", "anthroheightcalc", "BMI"]
    )

    # Join the covariates to the brain features

    mri_all_features_cov = mri_all_features.join(
        covariates,
        how="left",
    ).dropna()

    logging.info(
        "Sample size with all imaging features and covariates, number = %d",
        mri_all_features_cov.shape[0],
    )

    # %% This section joins the trajectory data

    dep_traj_path = Path(
        analysis_data_path,
        "ABCD_BPM_4Trajectories_long.txt",
    )

    dep_traj = pd.read_csv(
        dep_traj_path,
        sep="\t",
    )

    # Ensure src_subject_id is unique with information on trajectory and class
    dep_traj = dep_traj.groupby("src_subject_id").agg(
        {"trajectory": "first", "class": "first"}
    )

    dep_traj = dep_traj.rename(columns={"class": "class_label"})

    # Remap tjrajectory to numeric values
    # old: (low - 2, increasing - 1, decreasing - 3, high - 4)
    # new: (low - 0, increasing - 1, decreasing - 2, high - 3)

    class_label_mapping = {
        2: 0,
        1: 1,
        3: 2,
        4: 3,
    }

    # Apply the mapping to the class_label column as int
    dep_traj["class_label"] = (
        dep_traj["class_label"].replace(class_label_mapping).astype(int)
    ).astype("category")

    logging.info(
        "Remapping depression trajectory class labels to numeric values is error-free, Checked"  # noqa: E501
    )

    # Drop trajectory column
    dep_traj = dep_traj.drop(columns=["trajectory"])

    t1w_all_cortical_features_cov_traj = mri_all_features_cov.join(
        dep_traj,
        how="left",
    ).dropna()

    logging.info(
        "Sample size with all imaging features, covariates and depression trajectories,"
        "number = %d",
        t1w_all_cortical_features_cov_traj.shape[0],
    )

    # Standardize the continuous variables

    logging.info("Standardizing the continuous variables")

    categorical_variables = [
        "demo_sex_v2",
        "img_device_label",
        "rel_family_id",
        "demo_comb_income_v2",
        "class_label",
        "site_id_l",
    ]

    for col in categorical_variables:
        if col in t1w_all_cortical_features_cov_traj.columns:
            t1w_all_cortical_features_cov_traj[col] = (
                t1w_all_cortical_features_cov_traj[col].astype("category")
            )

    logging.info(
        "Make sure the following columns are categorical and excluded from standardisation: "  # noqa: E501
    )
    logging.info(", ".join(categorical_variables))

    # Get columns to scale (everything else)
    # BMI is already standardized, so it is not included in the scaling

    exclude_cols = categorical_variables

    logging.info(
        "Excluding the following columns from standardisation: %s",
        ", ".join(exclude_cols),
    )

    cols_to_scale = [
        col
        for col in t1w_all_cortical_features_cov_traj.columns
        if col not in exclude_cols
    ]

    # Standardize selected columns
    scaler = StandardScaler()

    t1w_all_cortical_features_cov_traj[cols_to_scale] = scaler.fit_transform(
        t1w_all_cortical_features_cov_traj[cols_to_scale]
    )

    logging.info("Standardization of continuous variables is error-free, Checked")

    rescaled_mri_all_features_with_traj = t1w_all_cortical_features_cov_traj.copy()

    # This is for performing analysis on left/right region separately
    # for regions of lateralised structures
    rescaled_mri_all_features_with_traj.to_csv(
        Path(
            processed_data_path,
            f"mri_all_features_with_traj_rescaled-{wave}.csv",
        ),
        index=True,
    )

    logging.info("Rescaled imaging features with traj saved to CSV")

    logging.info(
        "Final Sample size for wave:%s %d",
        wave,
        rescaled_mri_all_features_with_traj.shape[0],
    )

    # %%

    ### Create long-form data for left and right hemisphere features
    # Identify left/right hemisphere columns

    logging.info("Creating long-form data for left and right hemisphere features")

    lh_columns = [
        col for col in rescaled_mri_all_features_with_traj.columns if col.endswith("lh")
    ]

    logging.info("Number of left hemisphere features: %d", len(lh_columns))

    rh_columns = [
        col for col in rescaled_mri_all_features_with_traj.columns if col.endswith("rh")
    ]

    logging.info("Number of right hemisphere features: %d", len(rh_columns))

    # Identify all other columns (covariates, unilateral features, Traj.)
    other_columns = [
        col
        for col in rescaled_mri_all_features_with_traj.columns
        if col not in lh_columns + rh_columns
    ]

    # Create left and right hemisphere datasets
    lh_data = rescaled_mri_all_features_with_traj[other_columns + lh_columns].copy()
    rh_data = rescaled_mri_all_features_with_traj[other_columns + rh_columns].copy()

    # Add a prefix to the imaging columns to avoid duplicates
    # Rename feature columns to remove hemisphere suffixes
    lh_data = lh_data.rename(
        columns={col: f"img_{col[:-2]}" for col in lh_columns if col.endswith("lh")}
    )

    rh_data = rh_data.rename(
        columns={col: f"img_{col[:-2]}" for col in rh_columns if col.endswith("rh")}
    )

    # Add hemisphere and subject ID
    lh_data["hemisphere"] = "Left"
    rh_data["hemisphere"] = "Right"

    # Concatenate into long-form
    long_data = pd.concat(
        [lh_data, rh_data],
        axis=0,
    )

    logging.info("Creating long-form data is error-free, Checked")

    # Save (index already captured in column)
    processed_data_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    long_data.to_csv(
        Path(
            processed_data_path,
            f"mri_all_features_with_traj_long_rescaled-{wave}.csv",
        ),
        index=True,
    )

    logging.info("Long-form imaging features with traj saved to CSV")

    # %%

    # Average the bilateral features

    logging.info("Averaging bilateral features")

    # Check if the two lists match by region (assuming 'lh'/'rh' are prefixes)
    lh_regions = [col.replace("lh", "") for col in lh_columns]
    rh_regions = [col.replace("rh", "") for col in rh_columns]

    if lh_regions == rh_regions:
        logging.info("Left and right hemisphere feature lists MATCH by region.")

    else:
        logging.warning(
            "Left and right hemisphere feature lists DO NOT MATCH by region."
        )

    def average_hemisphere_columns(df, lh_columns, rh_columns, other_columns):
        avg_cols = {
            f"img_{lh.replace('lh', '')}": (df[lh] + df[rh]) / 2
            for lh, rh in zip(lh_columns, rh_columns)
        }
        other_cols = df[other_columns]
        return pd.concat([pd.DataFrame(avg_cols), other_cols], axis=1)

    rescaled_mri_all_features_with_traj_avg = average_hemisphere_columns(
        rescaled_mri_all_features_with_traj,
        lh_columns,
        rh_columns,
        other_columns,
    )

    logging.info("Averaging bilateral features is error-free, Checked")

    # Save the averaged features
    rescaled_mri_all_features_with_traj_avg.to_csv(
        Path(
            processed_data_path,
            f"mri_all_features_with_traj_avg_rescaled-{wave}.csv",
        ),
        index=True,
    )

    logging.info("Averaged imaging features with traj saved to CSV")

    # %%
    ### Now select the columns that are the phenotypes of interest for each modality

    logging.info("Selecting features of interest for each modality")

    ### Remove global features for all modality
    logging.info("Removing global features for each modality")

    logging.info("Cortical thickness global features:")
    logging.info("%s", list(t1w_cortical_thickness_pass.columns[-3:]))

    t1w_cortical_thickness_rois = list(t1w_cortical_thickness_pass.columns[1:-3])

    # For cortical volume

    logging.info("Cortical volume global features:")
    logging.info("%s", list(t1w_cortical_volume_pass.columns[-3:]))
    t1w_cortical_volume_rois = list(t1w_cortical_volume_pass.columns[1:-3])

    # For surface area

    logging.info("Cortical surface area global features:")
    logging.info("%s", list(t1w_cortical_surface_area_pass.columns[-3:]))
    t1w_cortical_surface_area_rois = list(t1w_cortical_surface_area_pass.columns[1:-3])

    ### For subcortical volume

    # NOTE: A list of global features selected by GPT, this might need to be updated

    global_subcortical_features = [
        "smri_vol_scs_csf",
        "smri_vol_scs_wholeb",
        "smri_vol_scs_intracranialv",
        "smri_vol_scs_latventricles",
        "smri_vol_scs_allventricles",
        "smri_vol_scs_subcorticalgv",
        "smri_vol_scs_suprateialv",
        "smri_vol_scs_wmhint",
    ]

    logging.info("Subcortical volume global features:")
    logging.info("%s", global_subcortical_features)

    # FA global features
    global_FA_features = [
        "FA_all_dti_atlas_tract_fibers",
        "FA_hemisphere_dti_atlas_tract_fibers_without_corpus_callosumrh",
        "FA_hemisphere_dti_atlas_tract_fibers_without_corpus_callosumlh",
        "FA_hemisphere_dti_atlas_tract_fibersrh",
        "FA_hemisphere_dti_atlas_tract_fiberslh",
    ]

    logging.info("FA global features:")
    logging.info("%s", global_FA_features)

    # MD global features
    global_MD_features = [
        "MD_all_dti_atlas_tract_fibers",
        "MD_hemisphere_dti_atlas_tract_fibers_without_corpus_callosumrh",
        "MD_hemisphere_dti_atlas_tract_fibers_without_corpus_callosumlh",
        "MD_hemisphere_dti_atlas_tract_fibersrh",
        "MD_hemisphere_dti_atlas_tract_fiberslh",
    ]

    logging.info("MD global features:")
    logging.info("%s", global_MD_features)

    # Step 2: Select subcortical ROIs
    t1w_subcortical_volume_rois = [
        col
        for col in t1w_subcortical_volume_pass.columns
        if col not in global_subcortical_features and col != "eventname"
    ]

    # For tract features

    FA_rois = [
        col
        for col in dmir_fractional_anisotropy_pass.columns
        if col not in global_FA_features and col != "eventname"
    ]

    MD_rois = [
        col
        for col in dmir_mean_diffusivity_pass.columns
        if col not in global_MD_features and col != "eventname"
    ]

    # Save features of interest for mixed effects models for each modalities

    def get_bilateral_and_unilateral_features(feature_list):
        """Returns bilateral and unilateral features from a list of features."""
        lh_roots = {f[:-2] for f in feature_list if f.endswith("lh")}
        rh_roots = {f[:-2] for f in feature_list if f.endswith("rh")}
        bilateral_roots = sorted(lh_roots & rh_roots)

        # Unilateral = present in only one hemisphere or has no suffix
        unilateral_features = [
            f for f in feature_list if (not f.endswith("lh") and not f.endswith("rh"))
        ]

        # Add prefix (img_) to the bilateral features

        bilateral_roots = [f"img_{f}" for f in bilateral_roots]

        return bilateral_roots, unilateral_features

    # Assemble all features for repeated effects modeling
    features_of_interest = {
        "bilateral_cortical_thickness": get_bilateral_and_unilateral_features(
            t1w_cortical_thickness_rois
        )[0],
        "bilateral_cortical_volume": get_bilateral_and_unilateral_features(
            t1w_cortical_volume_rois
        )[0],
        "bilateral_cortical_surface_area": get_bilateral_and_unilateral_features(
            t1w_cortical_surface_area_rois
        )[0],
        "bilateral_subcortical_volume": get_bilateral_and_unilateral_features(
            t1w_subcortical_volume_rois
        )[0],
        # Unilateral features are for performing GLM
        "unilateral_subcortical_features": get_bilateral_and_unilateral_features(
            t1w_subcortical_volume_rois
        )[1],
        "bilateral_tract_FA": get_bilateral_and_unilateral_features(FA_rois)[0],
        "bilateral_tract_MD": get_bilateral_and_unilateral_features(MD_rois)[0],
        # Unilateral features are for performing GLM
        "unilateral_tract_FA": get_bilateral_and_unilateral_features(FA_rois)[1],
        "unilateral_tract_MD": get_bilateral_and_unilateral_features(MD_rois)[1],
    }

    logging.info(
        "Creating features of interest for repeated effects modeling is error-free, Checked"  # noqa: E501
    )

    logging.info("Number of features for each modality:")
    for modality, features in features_of_interest.items():
        logging.info(f"{modality}: {len(features)} features")

    features_for_repeated_effects_path = Path(
        processed_data_path,
        "features_of_interest.json",
    )

    with open(features_for_repeated_effects_path, "w") as f:
        json.dump(features_of_interest, f)


if __name__ == "__main__":
    all_img_waves = [
        "baseline_year_1_arm_1",
        "2_year_follow_up_y_arm_1",
        "4_year_follow_up_y_arm_1",
    ]

    # Process all waves
    for wave in all_img_waves:
        preprocess(wave=wave)
