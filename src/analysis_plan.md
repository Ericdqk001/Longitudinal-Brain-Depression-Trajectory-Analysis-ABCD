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

- **DO NOT** standardize within waves (destroys temporal signal)
- Combine all waves into long format first
- Standardize across ALL time points using complete cases
- Preserves between-wave changes needed for slope estimation

**Missing Data Handling**:

- Keep subjects with ≥1 datapoint (mixed-effects can estimate from partial data)
- For standardization: Use all available non-missing values across all waves
- Apply standardization parameters to all data (including partial cases)

### Step 1: Mixed-Effects Modeling

For each brain region separately:

- **Model**: `brain_feature ~ time + sex + age_baseline + site + (1 + time | subject)`
- **Fixed effects**: Linear time trend, sex, baseline age, imaging site
- **Random effects**: Subject-specific intercept and slope
- **Extract**: Individual intercept (baseline) and slope (change rate) estimates

### Step 2: Trajectory Classification

For each brain region separately:

- **Model**: Multinomial logistic regression
- **Predictors**: Individual intercept and slope from Step 1
- **Outcome**: 4-class trajectory membership
- **Interpretation**: How baseline brain structure and rate of change associate with depression trajectories

## Statistical Considerations

- **Multiple comparisons**: Correct within imaging modalities (structural, DTI)
- **Analysis scope**: All available brain regions
- **Approach**: Exploratory - no specific regional hypotheses

## Expected Outputs

- Individual-level intercept/slope estimates for each brain region
- Association results between brain parameters and trajectory classes
- Effect sizes and confidence intervals for significant associations
