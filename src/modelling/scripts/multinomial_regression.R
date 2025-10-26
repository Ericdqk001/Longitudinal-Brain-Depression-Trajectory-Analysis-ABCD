# Load required libraries
# These are R packages we need for the analysis
library(data.table)  # Fast data manipulation (like polars in Python)
library(mclogit)     # For multinomial mixed effects models
library(jsonlite)    # For reading JSON files (like your features_of_interest.json)

# Set up logging function
# This creates a simple logging function to print messages with timestamps
log_info <- function(message) {
  cat("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] INFO: ", message, "\n", sep = "")
}

# Arguments as variables (matching your Python structure)
# These are the same path variables you use in Python for local testing
data_store_path <- file.path("/", "Volumes", "GenScotDepression")
version_name <- "dev"
experiment_number <- 1

# Build paths step by step (same as Python)
# file.path() is R's equivalent to Path() in Python - it builds file paths
analysis_root_path <- file.path(data_store_path, "users", "Eric",
                               "depression_trajectories")
version_path <- file.path(analysis_root_path, version_name)
experiments_path <- file.path(version_path, "experiments", paste0("exp_", experiment_number))
processed_data_path <- file.path(experiments_path, "processed_data")
results_path <- file.path(experiments_path, "results")

# Print paths to verify they're correct
log_info(paste("Analysis root path:", analysis_root_path))
log_info(paste("Processed data path:", processed_data_path))
log_info(paste("Results path:", results_path))

# Load brain slopes data (output from mixed_effect.py)
# file.path() builds the full path, like Path() in Python
slopes_path <- file.path(results_path, "subject_brain_slopes.csv")
log_info(paste("Loading slopes from:", slopes_path))

# fread() is data.table's fast CSV reader (like pl.read_csv() in Python)
# It automatically detects headers and data types
slopes_df <- fread(slopes_path)

# Print basic info about the data
log_info(paste("Slopes data shape:", nrow(slopes_df), "rows,", ncol(slopes_df), "columns"))
log_info(paste("Number of subjects:", length(unique(slopes_df$src_subject_id))))

# Load the long-format imaging data
# This is the same file that mixed_effect.py uses
data_path <- file.path(processed_data_path, "mri_all_features_cov_long_standardized_avg.csv")
log_info(paste("Loading data from:", data_path))

# Load the long-format data
data <- fread(data_path)

# Print basic info about the long-format data
log_info(paste("Data shape:", nrow(data), "rows,", ncol(data), "columns"))
log_info(paste("Number of subjects:", length(unique(data$src_subject_id))))

# Filter to baseline data only
# In R, we use data[condition] to filter rows
baseline_data <- data[eventname == "baseline_year_1_arm_1"]

# Print info about baseline data
log_info(paste("Baseline data shape:", nrow(baseline_data), "rows,",
               ncol(baseline_data), "columns"))
log_info(paste("Number of baseline subjects:",
               length(unique(baseline_data$src_subject_id))))

# Join slopes with baseline data
# In data.table, we use merge() or X[Y] syntax for joins
# This is an inner join on src_subject_id
merged_df <- merge(slopes_df, baseline_data, by = "src_subject_id")

# Print info about merged data
log_info(paste("Merged data shape:", nrow(merged_df), "rows,",
               ncol(merged_df), "columns"))
log_info(paste("Number of merged subjects:",
               length(unique(merged_df$src_subject_id))))

# Load features of interest from JSON file
features_path <- file.path(processed_data_path, "features_of_interest.json")
log_info(paste("Loading features of interest from:", features_path))

# Read JSON file using jsonlite
features_of_interest <- fromJSON(features_path)

# Print features info
log_info("Features loaded for modalities:")
for (modality in names(features_of_interest)) {
  n_features <- length(features_of_interest[[modality]])
  log_info(paste("Modality", modality, ":", n_features, "features"))
}

# Start multinomial regression analysis
log_info("Starting multinomial mixed effects modeling using mclogit...")

# Initialize storage for results
model_results <- list()
feature_importance <- data.table()

# Convert class_label to factor for modeling
merged_df$class_label <- as.factor(merged_df$class_label)

# Print trajectory class levels
log_info(paste("Trajectory classes:", paste(levels(merged_df$class_label), collapse = ", ")))

# Get total number of features across all modalities for progress tracking
total_features <- sum(sapply(features_of_interest, length))
feature_count <- 0

log_info(paste("Total features to process:", total_features))

test_modality <- names(features_of_interest)[1]

test_feature <- features_of_interest[[test_modality]][1]

log_info(paste("Testing model with:", test_feature, "from modality:", test_modality))

# Get column names
baseline_col <- test_feature
slope_col <- paste0(test_feature, "_slope")

# Check they exist
log_info(paste("Baseline column:", baseline_col, "- exists:", baseline_col %in% names(merged_df)))
log_info(paste("Slope column:", slope_col, "- exists:", slope_col %in% names(merged_df)))

# Prepare data for modeling
# Define covariates to include in the model
covariates <- c("demo_sex_v2", "demo_comb_income_v2",
"BMI_zscore", "interview_age", "age2", "img_device_label")

# Remove rows with missing values in key columns
required_cols <- c("class_label", "site_id_l", baseline_col, slope_col,
                   covariates)
complete_data <- merged_df[complete.cases(merged_df[, ..required_cols])]

log_info(paste("Complete cases:", nrow(complete_data), "out of",
               nrow(merged_df)))
log_info(paste("Number of sites:", length(unique(complete_data$site_id_l))))

# Check sample sizes
log_info("Sample sizes by trajectory class:")
print(table(complete_data$class_label))

# Create formula with baseline, slope, and covariates
# Use site-only random effects (family not feasible with singletons)
covariate_factorised <- c("factor(demo_sex_v2)", "factor(demo_comb_income_v2)", "interview_age", "factor(img_device_label)")

covariate_str <- paste(covariate_factorised, collapse = " + ")

formula_str <- paste("class_label ~", baseline_col, "+", slope_col,
                     "+", covariate_str)
log_info(paste("Formula:", formula_str))

# Convert to formula object
model_formula <- as.formula(formula_str)

log_info("Note: Using site-only random effects (1|site_id_l)")
log_info("Family random effects not included - most families have only 1 child (median=1)")

# Fit the model using mblogit (multinomial logit with random effects)
tryCatch({
  log_info("Starting model fitting...")
  model <- mblogit(
    formula = model_formula,
    data = complete_data,
    # random = ~ 1 | site_id_l  # Site-level random effects only
  )

  log_info("Model fitting completed successfully!")

  # Display model summary
  log_info("Model summary:")
  model_summary <- summary(model)
  print(model_summary)

  # Store results
  model_results[[test_feature]] <- model

  # ===== TEST: Explore model structure to find coefficients, SEs, p-values =====
  log_info("===== EXPLORING MODEL STRUCTURE =====")

  # Check what's in the model object
  log_info("Model class:")
  print(class(model))

  log_info("Model names (top-level components):")
  print(names(model))

  # Check what's in the summary object
  log_info("Summary class:")
  print(class(model_summary))

  log_info("Summary names (top-level components):")
  print(names(model_summary))

  # Extract coefficients table (usually contains estimates, SEs, z-values, p-values)
  log_info("Extracting coefficient table from summary:")

  # Try different common names for coefficient tables
  if ("coefficients" %in% names(model_summary)) {
    coef_table <- model_summary$coefficients
    log_info("Found 'coefficients' in summary")
  } else if ("coef" %in% names(model_summary)) {
    coef_table <- model_summary$coef
    log_info("Found 'coef' in summary")
  } else {
    log_info("Trying coef() function:")
    coef_table <- coef(model_summary)
  }

  log_info("Coefficient table structure:")
  print(str(coef_table))

  log_info("Coefficient table (first few rows):")
  print(head(coef_table, 20))

  log_info("Coefficient table column names:")
  print(colnames(coef_table))

  log_info("Coefficient table row names:")
  print(rownames(coef_table))

  # ===== TEST: Extract specific coefficients for our features of interest =====
  log_info("===== EXTRACTING FEATURE-SPECIFIC RESULTS =====")

  # Row names have format: "1~predictor", "2~predictor", "3~predictor"
  # We directly use row names to access coefficients (more robust)

  # Construct row names for baseline and slope for each class
  baseline_class1_name <- paste0("1~", baseline_col)
  baseline_class2_name <- paste0("2~", baseline_col)
  baseline_class3_name <- paste0("3~", baseline_col)

  slope_class1_name <- paste0("1~", slope_col)
  slope_class2_name <- paste0("2~", slope_col)
  slope_class3_name <- paste0("3~", slope_col)

  # Check if these row names exist in the coefficient table
  log_info("Checking for baseline coefficient rows:")
  log_info(paste("  Class 1:", baseline_class1_name,
                 "exists:", baseline_class1_name %in% rownames(coef_table)))
  log_info(paste("  Class 2:", baseline_class2_name,
                 "exists:", baseline_class2_name %in% rownames(coef_table)))
  log_info(paste("  Class 3:", baseline_class3_name,
                 "exists:", baseline_class3_name %in% rownames(coef_table)))

  log_info("Checking for slope coefficient rows:")
  log_info(paste("  Class 1:", slope_class1_name,
                 "exists:", slope_class1_name %in% rownames(coef_table)))
  log_info(paste("  Class 2:", slope_class2_name,
                 "exists:", slope_class2_name %in% rownames(coef_table)))
  log_info(paste("  Class 3:", slope_class3_name,
                 "exists:", slope_class3_name %in% rownames(coef_table)))

  # ===== TEST: Create results storage format =====
  log_info("===== TESTING RESULTS STORAGE FORMAT =====")

  # Extract values for each class comparison using row names directly
  # Columns: Estimate, Std. Error, z value, Pr(>|z|)

  # Create results row with all coefficients
  results_row <- data.table(
    imaging_feature = test_feature,
    modality = test_modality,

    # Baseline - Class 1 vs 0
    baseline_coef_class1 = coef_table[baseline_class1_name, "Estimate"],
    baseline_se_class1 = coef_table[baseline_class1_name, "Std. Error"],
    baseline_z_class1 = coef_table[baseline_class1_name, "z value"],
    baseline_p_class1 = coef_table[baseline_class1_name, "Pr(>|z|)"],

    # Baseline - Class 2 vs 0
    baseline_coef_class2 = coef_table[baseline_class2_name, "Estimate"],
    baseline_se_class2 = coef_table[baseline_class2_name, "Std. Error"],
    baseline_z_class2 = coef_table[baseline_class2_name, "z value"],
    baseline_p_class2 = coef_table[baseline_class2_name, "Pr(>|z|)"],

    # Baseline - Class 3 vs 0
    baseline_coef_class3 = coef_table[baseline_class3_name, "Estimate"],
    baseline_se_class3 = coef_table[baseline_class3_name, "Std. Error"],
    baseline_z_class3 = coef_table[baseline_class3_name, "z value"],
    baseline_p_class3 = coef_table[baseline_class3_name, "Pr(>|z|)"],

    # Slope - Class 1 vs 0
    slope_coef_class1 = coef_table[slope_class1_name, "Estimate"],
    slope_se_class1 = coef_table[slope_class1_name, "Std. Error"],
    slope_z_class1 = coef_table[slope_class1_name, "z value"],
    slope_p_class1 = coef_table[slope_class1_name, "Pr(>|z|)"],

    # Slope - Class 2 vs 0
    slope_coef_class2 = coef_table[slope_class2_name, "Estimate"],
    slope_se_class2 = coef_table[slope_class2_name, "Std. Error"],
    slope_z_class2 = coef_table[slope_class2_name, "z value"],
    slope_p_class2 = coef_table[slope_class2_name, "Pr(>|z|)"],

    # Slope - Class 3 vs 0
    slope_coef_class3 = coef_table[slope_class3_name, "Estimate"],
    slope_se_class3 = coef_table[slope_class3_name, "Std. Error"],
    slope_z_class3 = coef_table[slope_class3_name, "z value"],
    slope_p_class3 = coef_table[slope_class3_name, "Pr(>|z|)"]
  )

  log_info("Results row created:")
  print(results_row)

  # Test saving to CSV locally
  test_output_path <- "test_multinomial_results.csv"
  fwrite(results_row, test_output_path)
  log_info(paste("Test results saved to:", test_output_path))
  log_info(paste("Full path:", normalizePath(test_output_path)))

  log_info("===== END OF MODEL STRUCTURE EXPLORATION =====")

}, error = function(e) {
  log_info(paste("Error fitting model for", test_feature, ":", e$message))
})
