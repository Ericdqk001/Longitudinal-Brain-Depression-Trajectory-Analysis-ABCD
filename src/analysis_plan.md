# Longitudinal Brain-Depression Trajectory Analysis Plan

## Objective

Investigate associations between brain structural changes over time and depressive trajectory classes in adolescents using ABCD Study data.

## Variables

- **Outcome**: Depressive trajectory classes (0=low, 1=increasing, 2=decreasing, 3=high)
- **Predictors**: Brain imaging features across 3 waves
  - Cortical thickness, cortical volume, surface area
  - Subcortical volume
  - Fractional anisotropy (FA), mean diffusivity (MD)
- **Hemisphere averaging**: Take mean of left/right hemisphere values to simplify analysis

## Analysis Pipeline

### Step 0: Image Preprocessing

**Data Structure**: Process each wave separately for QC/cleaning, then combine into long format with time column for mixed-effects modeling.

**Standardization Strategy**:

- **Baseline standardization**: Use Wave 1 (baseline) mean/std parameters for all waves
- Rationale: Preserves temporal trajectories while using least biased distribution
- Later waves with systematic dropout would bias pooled standardization

**Sample Selection**:

- **Require baseline data**: Essential for standardization reference and interpretation
- Allow subjects with ≥1 additional timepoint for trajectory estimation
- Mixed-effects models handle remaining missing data under MAR assumptions

### Step 1: Mixed-Effects Modeling

For each brain region separately:

- **Model**: `brain_feature ~ time + sex + age_baseline + site + SES + baseline_depression + (1 + time | subject)`
- **Fixed effects**: Linear time trend, sex, baseline age, imaging site, dropout predictors
- **Random effects**: Subject-specific intercept and slope
- **Dropout predictors**: Include SES, baseline depression to make MAR more plausible
- **Extract**: Individual intercept (baseline) and slope (change rate) estimates

### Step 2: Trajectory Classification

For each brain region separately:

- **Model**: Multinomial logistic regression
- **Predictors**: Individual intercept and slope from Step 1
- **Outcome**: 4-class trajectory membership
- **Interpretation**: How baseline brain structure and rate of change associate with depression trajectories

### Step 3: Missing Data Sensitivity Analysis

**MAR Assumption**: Since MAR is untestable, conduct sensitivity analyses:

- **Missingness patterns**: Examine systematic dropout by demographics, baseline measures
- **Multiple imputation**: Compare results with different imputation strategies
- **Pattern-mixture models**: Test if conclusions change under MNAR assumptions
- **Complete case analysis**: Compare with baseline-only subjects as robustness check

## Statistical Considerations

- **Multiple comparisons**: Correct within imaging modalities (structural, DTI)
- **Analysis scope**: All available brain regions
- **Approach**: Exploratory - no specific regional hypotheses

## Expected Outputs

- Individual-level intercept/slope estimates for each brain region
- Association results between brain parameters and trajectory classes
- Effect sizes and confidence intervals for significant associations
