# Supervisor Meeting Questions

## 1. Standardization Method

**Question**: We propose baseline standardization (using Wave 1 mean/std for all waves) rather than pooled standardization. Rationale: (a) preserves temporal trajectories for mixed-effects modeling, (b) avoids bias from systematic dropout in later waves. Do you agree with this approach?

**Alternative considered**: Pooled standardization across all timepoints, but this would be biased by missing data patterns.

## 2. Baseline Data Requirement

**Question**: Should we require all subjects to have baseline brain imaging data? This is necessary for standardization consistency and interpretability, but may reduce sample size.

**Trade-off**: Statistical necessity vs. sample size preservation.

## 3. Sample Definition for Mixed-Effects Step

**Question**: For Step 1 (mixed-effects modeling), should we restrict to subjects who also have trajectory class data, or include all subjects with brain imaging regardless of trajectory availability?

**Option A**: Restrict to subjects with both brain + trajectory data

- Pro: Single consistent sample across both analysis steps
- Con: Smaller sample, potentially less generalizable brain parameter estimates

**Option B**: Include all subjects with brain data in Step 1

- Pro: Larger sample for brain parameter estimation, more generalizable
- Con: Two different samples to describe, Step 2 uses subset of Step 1

**Implication**: Option B requires describing two samples in methods but may yield more robust brain estimates.
