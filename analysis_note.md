# Analysis Decisions

## 1. Preprocessing (`src/preprocess/preprocess.py`)

### Standardization Strategy

- **Baseline-anchored standardization** (lines 970-991): All imaging features are standardized using mean and standard deviation parameters calculated from baseline (wave 1) data only
- These baseline parameters are then applied to standardize features at all timepoints (baseline, 2-year, 4-year follow-up)
- Rationale: Ensures consistent scaling across waves and prevents information leakage from future timepoints

### Subject Inclusion Criteria

- **Dropped subjects without follow-up data** (lines 1002-1024): Subjects with only baseline data were excluded from the final dataset
- Rationale: Mixed-effects models require repeated measurements to estimate individual slopes. Subjects with single timepoint provide minimal information for slope estimation and can cause convergence issues

### Hemisphere Averaging

- **Bilateral feature averaging** (lines 1090-1135): For all bilateral brain features (those with 'lh' and 'rh' suffixes), the left and right hemisphere values were averaged to create a single feature per region
- Averaged features are prefixed with `img_` to distinguish them from original hemisphere-specific features
- Rationale: Simplifies analysis by reducing feature dimensionality and focusing on overall regional effects rather than hemisphere-specific patterns

## 2. Mixed-Effects Modeling (`src/modelling/scripts/mixed_effect.py`)

### Random Effects Structure

- **Formula** (line 130): `(0 + time|src_subject_id) + (1|site_id_l/rel_family_id)`

#### Random Slopes Only for Subjects

- `(0 + time|src_subject_id)`: Random slopes for time by subject, with **no random intercept**
- Rationale:
  - The dataset has limited repeated measurements per subject (maximum 3 timepoints over 4 years)
  - Estimating both random intercepts and random slopes would require too many random effects parameters relative to the amount of data available per subject
  - Random slopes capture individual variation in brain development trajectories, which is the primary quantity of interest for this analysis
  - Fixed effects already account for baseline differences through the inclusion of baseline covariates

#### Nested Random Effects for Site and Family

- `(1|site_id_l/rel_family_id)`: Nested random intercepts for site and family within site
- Site-level random intercepts account for systematic differences across scanning sites (scanner protocols, demographics, etc.)
- Family-level random intercepts (nested within site) account for genetic and shared environmental factors among related individuals
- Note: For subsequent multinomial regression, family random effects were removed due to most families being singletons (86% have only 1 child), making family-level variance estimation infeasible

## 3. Multinomial Regression (`src/modelling/scripts/multinomial_regression.R`)

### Sample Characteristics

- **Total baseline subjects**: 6127
- **Trajectory class distribution**:
  - Class 0 (Low): 4819 subjects (78.7%)
  - Class 1 (Increasing): 477 subjects (7.8%)
  - Class 2 (Decreasing): 501 subjects (8.2%)
  - Class 3 (High): 330 subjects (5.4%)

### Random Effects Structure

- **No random effects used**: Multinomial logistic regression fitted as fixed-effects only model
- Rationale for excluding random effects:
  - **Family random effects infeasible**: Most families are singletons (median = 1 child per family). Random effects require within-group variation to estimate group-level variance, which is absent when 86% of families contribute only one subject
  - **Site random effects cause non-convergence**: Including site-level random effects `(1|site_id_l)` in the multinomial model prevented algorithm convergence, likely due to the complexity of estimating random effects across multiple outcome categories with sparse data in some trajectory classes

### Model Specification

- **Predictors**: Baseline brain feature + individual slope (from mixed-effects model) + covariates
- **Covariates**: BMI_zscore, demo_sex_v2, demo_comb_income_v2, interview_age, age2, img_device_label
- **Outcome**: 4-level depression trajectory class (0=Low, 1=Increasing, 2=Decreasing, 3=High)
