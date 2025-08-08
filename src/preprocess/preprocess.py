import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess(
    data_store_path: Path,
    version_name: str = "dev",
    experiment_number: int = 1,
):
    """Preprocess longitudinal neuroimaging data for mixed-effects modeling.

    Processes ABCD Study data across three waves (baseline, 2-year, 4-year follow-up)
    to create standardized long-format dataset ready for mixed-effects analysis:

    1. Baseline standardization: Uses wave 1 parameters for all timepoints
    2. Long-format creation: Concatenates waves with time variable for mixed-effects
    3. Device encoding: Consistent label encoding across waves for
    time-varying covariates

    Data processing includes:
        - Neuroimaging features: Cortical thickness/volume/area, subcortical volume,
        DTI (FA/MD)
        - Quality control: T1w/dMRI inclusion criteria, clinical report screening
        - Covariates: Demographics, anthropometric measures, site/device information
        - Missing data: Preserves all subjects with complete baseline data;
        follow-ups filtered to baseline subjects only

    Saves:
        Individual wave files and final long-format standardized dataset to
        processed_data_path.
    """
    all_waves = [
        "baseline_year_1_arm_1",
        "2_year_follow_up_y_arm_1",
        "4_year_follow_up_y_arm_1",
    ]

    logging.info("Starting preprocessing")

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

    for wave in all_waves:
        logging.info("-----------------------")
        logging.info("Processing wave: %s", wave)
        # %%

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
        demographics_bl = demographics[
            demographics.eventname == "baseline_year_1_arm_1"
        ]

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

        # First, we apply quality control to T1 weighted images (
        # for structural features).
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

        # Select the data for the subjects who passed the quality control

        # Cortical thickness data
        mri_y_smr_thk_dst = mri_y_smr_thk_dst[mri_y_smr_thk_dst.eventname == wave]

        logging.info(
            "Sample size with T1w cortical thickness data, number = %d, wave = %s",
            mri_y_smr_thk_dst.shape[0],
            wave,
        )

        t1w_cortical_thickness_pass = mri_y_smr_thk_dst[
            mri_y_smr_thk_dst.index.isin(subs_pass.index)
        ]

        logging.info(
            "Sample size of CT data after QC, number = %d, wave = %s",
            t1w_cortical_thickness_pass.shape[0],
            wave,
        )

        # Cortical volume data
        mri_y_smr_vol_dst = mri_y_smr_vol_dst[mri_y_smr_vol_dst.eventname == wave]

        logging.info(
            "Sample size with T1w cortical volume data, number = %d, wave = %s",
            mri_y_smr_vol_dst.shape[0],
            wave,
        )

        t1w_cortical_volume_pass = mri_y_smr_vol_dst[
            mri_y_smr_vol_dst.index.isin(subs_pass.index)
        ]

        logging.info(
            "Sample size of CV data after QC, number = %d, wave = %s",
            t1w_cortical_volume_pass.shape[0],
            wave,
        )

        # Cortical surface area data
        mri_y_smr_area_dst = mri_y_smr_area_dst[mri_y_smr_area_dst.eventname == wave]

        logging.info(
            "Sample size with T1w cortical surface area data, number = %d, wave = %s",
            mri_y_smr_area_dst.shape[0],
            wave,
        )

        t1w_cortical_surface_area_pass = mri_y_smr_area_dst[
            mri_y_smr_area_dst.index.isin(subs_pass.index)
        ]

        logging.info(
            "Sample size of SA data after QC, number = %d, wave = %s",
            t1w_cortical_surface_area_pass.shape[0],
            wave,
        )

        # Subcortical volume

        t1w_subcortical_volume = mri_y_smr_vol_aseg[
            mri_y_smr_vol_aseg.eventname == wave
        ]

        logging.info(
            "Sample size with T1w subcortical volume data, number = %d, wave = %s",
            t1w_subcortical_volume.shape[0],
            wave,
        )

        t1w_subcortical_volume_pass = t1w_subcortical_volume[
            t1w_subcortical_volume.index.isin(subs_pass.index)
        ]

        # NOTE: These columns were dropped because they had all missing values
        # or all zeros

        subcortical_all_zeros_cols = [
            "smri_vol_scs_lesionlh",
            "smri_vol_scs_lesionrh",
            "smri_vol_scs_wmhintlh",
            "smri_vol_scs_wmhintrh",
        ]

        t1w_subcortical_volume_pass = t1w_subcortical_volume_pass.drop(
            columns=subcortical_all_zeros_cols
        )

        logging.info("Subcortical all zeros columns dropped")
        logging.info("Column names: %s", subcortical_all_zeros_cols)

        logging.info(
            "Sample size of subcortical volume data after QC, number = %d, wave = %s",
            t1w_subcortical_volume_pass.shape[0],
            wave,
        )

        # # Add tracts data (mri_y_dti_fa_fs_at (FA), mri_y_dti_md_fs_at(MD))

        dmir_fractional_anisotropy = mri_y_dti_fa_fs_at[
            mri_y_dti_fa_fs_at.eventname == wave
        ]

        logging.info(
            "Sample size with dMRI fractional anisotropy data, number = %d, wave = %s",
            dmir_fractional_anisotropy.shape[0],
            wave,
        )

        # Dropna later because some columns will be removed
        dmir_fractional_anisotropy_pass = dmir_fractional_anisotropy[
            dmir_fractional_anisotropy.index.isin(subs_pass.index)
        ]

        logging.info(
            "Sample size of FA data after QC, number = %d, wave = %s",
            dmir_fractional_anisotropy_pass.shape[0],
            wave,
        )

        dmir_mean_diffusivity = mri_y_dti_md_fs_at[mri_y_dti_md_fs_at.eventname == wave]

        logging.info(
            "Sample size with dMRI mean diffusivity data, number = %d, wave = %s",
            dmir_mean_diffusivity.shape[0],
            wave,
        )

        dmir_mean_diffusivity_pass = dmir_mean_diffusivity[
            dmir_mean_diffusivity.index.isin(subs_pass.index)
        ]

        logging.info(
            "Sample size of MD data after QC, number = %d, wave = %s",
            dmir_mean_diffusivity_pass.shape[0],
            wave,
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

        # Sort the columns of the DTI dataframes because they can cause issues with
        # later concatenation to create long form data
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

        dmir_fractional_anisotropy_pass = sort_dmri_columns(
            dmir_fractional_anisotropy_pass
        )

        dmir_mean_diffusivity_pass = sort_dmri_columns(dmir_mean_diffusivity_pass)

        logging.info("Sort dMRI FA/MD columns by number at the end of each column name")
        logging.info("For example: 'dmdtifp1_43', ''dmdtifp1_44', 'dmdtifp1_45'")
        logging.info("Sorting is error-free, checked")

        # Parse the DTI features
        dti_features_mapping = parse_dti_features_pretty(dmri_data_dict)

        logging.info(
            "Parsing FA/MD feature descriptions to new feature names is error-free, Checked"  # noqa: E501
        )

        # Rename the columns in dmri data
        dmir_fractional_anisotropy_pass.rename(
            columns=dti_features_mapping,
            inplace=True,
        )

        logging.info("Renaming FA features to new feature names is error-free, Checked")

        # Drop these columns because they are duplicates with a
        # slightly different regional focus
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

        # dmir_fractional_anisotropy_pass = dmir_fractional_anisotropy_pass.dropna()

        # logging.info(
        #     "Sample size with complete FA data after QC, number = %d",
        #     dmir_fractional_anisotropy_pass.shape[0],
        # )

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

        # dmir_mean_diffusivity_pass = dmir_mean_diffusivity_pass.dropna()

        # logging.info(
        #     "Sample size with complete MD data after QC, number = %d",
        #     dmir_mean_diffusivity_pass.shape[0],
        # )

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
            "Sample size with all imaging features (missing values included), number = %d",  # noqa: E501
            mri_all_features.shape[0],
        )

        # %%

        # Drop eventname column
        mri_all_features = mri_all_features.drop(columns="eventname")

        ### Add covariates to be considered in the analysis

        logging.info("Adding covariates to the imaging features")

        # For imaging device ID
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

        # Not available category (777:refused to answer, 999: don't know, missing values) # noqa: E501

        household_income = household_income.replace(
            [777, 999],
            np.nan,
        )

        logging.info(
            "Subjects who either refused to answer or don't know their income are set to NA"  # noqa: E501
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
            demographics_bl.demo_sex_v2,  # Baseline info
            mri_y_adm_info.mri_info_deviceserialnumber,
            abcd_y_lt.interview_age,
            abcd_y_lt.age2,
            household_income,  # Baseline info
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
            columns=[
                "weight_kg",
                "height_m",
                "anthroweightcalc",
                "anthroheightcalc",
                "BMI",
            ]
        )

        # Join the covariates to the brain features

        mri_all_features_cov = mri_all_features.join(
            covariates,
            how="inner",
        )

        logging.info(
            "Sample size with all imaging features and covariates (with no missing values), number = %d, wave = %s",  # noqa: E501
            mri_all_features_cov.shape[0],
            wave,
        )

        # Save the processed data

        mri_all_features_cov.to_csv(
            Path(
                processed_data_path,
                f"mri_all_features_cov_{wave}.csv",
            ),
            index=True,
        )

        logging.info(
            "Processed data with all imaging features and covariates (with no missing values) saved to %s",  # noqa: E501
            processed_data_path,
        )

    # %% This section joins all waves of the processed data together

    # Load each wave explicitly by name
    baseline_data = pd.read_csv(
        Path(processed_data_path, "mri_all_features_cov_baseline_year_1_arm_1.csv"),
        index_col=0,
        low_memory=False,
    )

    wave2_data = pd.read_csv(
        Path(processed_data_path, "mri_all_features_cov_2_year_follow_up_y_arm_1.csv"),
        index_col=0,
        low_memory=False,
    )

    wave3_data = pd.read_csv(
        Path(processed_data_path, "mri_all_features_cov_4_year_follow_up_y_arm_1.csv"),
        index_col=0,
        low_memory=False,
    )

    # Join the trajectories data before concatenation

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

    # Remap trajectory to numeric values
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

    baseline_data_traj = baseline_data.join(
        dep_traj,
        how="left",
    ).dropna()

    wave2_data_traj = wave2_data.join(
        dep_traj,
        how="left",
    ).dropna()

    wave3_data_traj = wave3_data.join(
        dep_traj,
        how="left",
    ).dropna()

    # Standardize data using baseline parameters before concatenation
    logging.info("Standardizing the continuous variables with baseline mean and std")

    categorical_variables = [
        "demo_sex_v2",
        "rel_family_id",
        "demo_comb_income_v2",
        "class_label",
        "site_id_l",
    ]

    # Add BMI_zscore and device ID (which will be encoded later) to exclude list
    exclude_cols = categorical_variables + ["BMI_zscore", "mri_info_deviceserialnumber"]

    logging.info(
        "Excluding the following columns from standardisation: %s",
        ", ".join(exclude_cols),
    )

    cols_to_scale = [
        col for col in baseline_data_traj.columns if col not in exclude_cols
    ]

    # Fit scaler on baseline data only
    scaler = StandardScaler()
    scaler.fit(baseline_data_traj[cols_to_scale])

    logging.info("Fitted scaler on baseline data")

    # Apply baseline scaler parameters to all waves
    baseline_data_traj_rescaled = baseline_data_traj.copy()
    baseline_data_traj_rescaled[cols_to_scale] = scaler.transform(
        baseline_data_traj[cols_to_scale]
    )

    wave2_data_traj_rescaled = wave2_data_traj.copy()
    wave2_data_traj_rescaled[cols_to_scale] = scaler.transform(
        wave2_data_traj[cols_to_scale]
    )

    wave3_data_traj_rescaled = wave3_data_traj.copy()
    wave3_data_traj_rescaled[cols_to_scale] = scaler.transform(
        wave3_data_traj[cols_to_scale]
    )

    logging.info("Applied baseline standardization parameters to all waves")

    # Get baseline subjects and filter out follow-ups who are not in baseline
    baseline_subjects = baseline_data_traj_rescaled.index
    wave2_traj_rescaled_filtered = wave2_data_traj_rescaled[
        wave2_data_traj_rescaled.index.isin(baseline_subjects)
    ]
    wave3_traj_rescaled_filtered = wave3_data_traj_rescaled[
        wave3_data_traj_rescaled.index.isin(baseline_subjects)
    ]

    # Add time variable to the dataframes
    baseline_data_traj_rescaled["time"] = 0
    wave2_traj_rescaled_filtered["time"] = 2
    wave3_traj_rescaled_filtered["time"] = 4

    # Concatenate into long format
    long_data = pd.concat(
        [
            baseline_data_traj_rescaled,
            wave2_traj_rescaled_filtered,
            wave3_traj_rescaled_filtered,
        ],
        axis=0,
    )

    logging.info(
        f"Baseline subjects with trajectory data sample size: {len(baseline_data_traj)}"
    )
    logging.info(
        f"2 year follow-up with trajectory data (filtered to baseline) sample size: {len(wave2_traj_rescaled_filtered)}"  # noqa: E501
    )
    logging.info(
        f"4 year follow-up with trajectory data (filtered to baseline) sample size: {len(wave3_traj_rescaled_filtered)}"  # noqa: E501
    )
    logging.info(f"Processed long data shape: {long_data.shape}")

    # Encode imaging device ID

    le = LabelEncoder()

    # Using .fit_transform function to fit label
    label = le.fit_transform(long_data["mri_info_deviceserialnumber"])

    long_data["img_device_label"] = label

    logging.info(
        "Using LabelEncoder to encode the imaging device ID is error-free, Checked"
    )

    # Add img_device_label to the categorical variables list
    categorical_variables.append("img_device_label")

    # Convert categorical variables to proper type
    for col in categorical_variables:
        if col in long_data.columns:
            long_data[col] = long_data[col].astype("category")

    logging.info("Standardization with baseline parameters completed")

    # Save the standardized long format data
    long_data.to_csv(
        Path(
            processed_data_path,
            "mri_all_features_cov_long_standardized.csv",
        ),
        index=True,
    )
