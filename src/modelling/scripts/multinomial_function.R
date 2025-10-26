# Multinomial Logistic Regression for Depression Trajectories
# Fits multinomial models for all brain features to predict trajectory class
# using baseline values and individual slopes from mixed-effects models

# Install required packages if not already installed
required_packages <- c("data.table", "mclogit", "jsonlite")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(paste("Installing package:", pkg, "\n"))
    install.packages(pkg, repos = "https://cran.rstudio.com/",
                     quiet = TRUE)
  }
}

# Load required libraries
library(data.table)  # Fast data manipulation
library(mclogit)     # For multinomial logistic regression
library(jsonlite)    # For reading JSON files

# Set up logging function
log_info <- function(message) {
  cat("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] INFO: ", message,
      "\n", sep = "")
}

# Main function for multinomial regression analysis
fit_multinomial_models <- function(data_store_path="/Volumes/GenScotDepression",
                                   version_name = "dev",
                                   experiment_number = 1) {

  log_info("Starting multinomial logistic regression analysis...")
  log_info(paste("Data store path:", data_store_path))
  log_info(paste("Version name:", version_name))
  log_info(paste("Experiment number:", experiment_number))

  # Build paths (matching Python structure)
  analysis_root_path <- file.path(data_store_path, "users", "eric",
                                  "depression_trajectories")
  version_path <- file.path(analysis_root_path, version_name)
  experiments_path <- file.path(version_path, "experiments",
                                paste0("exp_", experiment_number))
  processed_data_path <- file.path(experiments_path, "processed_data")
  results_path <- file.path(experiments_path, "results")

  log_info(paste("Results path:", results_path))

  # Load brain slopes data (output from mixed_effect.py)
  slopes_path <- file.path(results_path, "subject_brain_slopes.csv")
  log_info(paste("Loading slopes from:", slopes_path))
  slopes_df <- fread(slopes_path)

  log_info(paste("Slopes data shape:", nrow(slopes_df), "rows,",
                 ncol(slopes_df), "columns"))
  log_info(paste("Number of subjects:",
                 length(unique(slopes_df$src_subject_id))))

  # Load the long-format imaging data
  data_path <- file.path(processed_data_path,
                         "mri_all_features_cov_long_standardized_avg.csv")
  log_info(paste("Loading data from:", data_path))
  data <- fread(data_path)

  # Filter to baseline data only
  # nolint start: object_usage_linter
  baseline_data <- data[eventname == "baseline_year_1_arm_1"]
  # nolint end

  log_info(paste("Baseline data shape:", nrow(baseline_data), "rows,",
                 ncol(baseline_data), "columns"))
  log_info(paste("Number of baseline subjects:",
                 length(unique(baseline_data$src_subject_id))))

  # Join slopes with baseline data
  merged_df <- merge(slopes_df, baseline_data, by = "src_subject_id")

  log_info(paste("Merged data shape:", nrow(merged_df), "rows,",
                 ncol(merged_df), "columns"))
  log_info(paste("Number of merged subjects:",
                 length(unique(merged_df$src_subject_id))))

  # Load features of interest from JSON file
  features_path <- file.path(processed_data_path, "features_of_interest.json")
  log_info(paste("Loading features of interest from:", features_path))
  features_of_interest <- fromJSON(features_path)

  log_info("Features loaded for modalities:")
  for (modality in names(features_of_interest)) {
    n_features <- length(features_of_interest[[modality]])
    log_info(paste("  Modality", modality, ":", n_features, "features"))
  }

  # Convert class_label to factor for modeling
  merged_df$class_label <- as.factor(merged_df$class_label)

  log_info(paste("Trajectory classes:",
                 paste(levels(merged_df$class_label), collapse = ", ")))

  # Print trajectory class distribution
  log_info("Trajectory class distribution:")
  print(table(merged_df$class_label))

  # Define covariates to include in the model
  covariates <- c("demo_sex_v2", "demo_comb_income_v2",
                  "BMI_zscore", "interview_age", "age2", "img_device_label")

  log_info(paste("Covariates:", paste(covariates, collapse = ", ")))

  # Factorize categorical covariates for formula
  covariate_factorised <- c("factor(demo_sex_v2)",
                            "factor(demo_comb_income_v2)",
                            "BMI_zscore",
                            "interview_age",
                            "age2",
                            "factor(img_device_label)")

  # Initialize storage for results
  results_list <- list()
  failed_models <- c()

  # Get total number of features across all modalities
  total_features <- sum(sapply(features_of_interest, length))
  feature_count <- 0

  log_info(paste("Total features to process:", total_features))
  log_info("Starting model fitting loop...")

  # Loop through all modalities and features
  for (modality in names(features_of_interest)) {
    log_info(paste("Processing modality:", modality))

    for (feature in features_of_interest[[modality]]) {
      feature_count <- feature_count + 1

      # Progress logging every 50 features
      if (feature_count %% 50 == 0) {
        log_info(paste("Progress:", feature_count, "/", total_features,
                       "features processed"))
      }

      # Get column names
      baseline_col <- feature
      slope_col <- paste0(feature, "_slope")

      # Check if columns exist
      if (!(baseline_col %in% names(merged_df))) {
        log_info(paste("Warning: Baseline column", baseline_col,
                       "not found, skipping"))
        failed_models <- c(failed_models, paste0(modality, ":", feature))
        next
      }

      if (!(slope_col %in% names(merged_df))) {
        log_info(paste("Warning: Slope column", slope_col,
                       "not found, skipping"))
        failed_models <- c(failed_models, paste0(modality, ":", feature))
        next
      }

      # Prepare data for modeling
      required_cols <- c("class_label", baseline_col, slope_col, covariates) # nolint
      # nolint start: object_usage_linter
      complete_data <- merged_df[complete.cases(merged_df[, ..required_cols])]
      # nolint end

      # Build formula
      covariate_str <- paste(covariate_factorised, collapse = " + ")
      formula_str <- paste("class_label ~", baseline_col, "+", slope_col,
                           "+", covariate_str)
      model_formula <- as.formula(formula_str)

      # Fit the model with error handling
      tryCatch({
        # Fit multinomial logistic regression (no random effects)
        model <- mblogit(
          formula = model_formula,
          data = complete_data
        )

        # Extract coefficient table
        model_summary <- summary(model)
        coef_table <- model_summary$coefficients

        # Construct row names for baseline and slope for each class
        baseline_class1_name <- paste0("1~", baseline_col)
        baseline_class2_name <- paste0("2~", baseline_col)
        baseline_class3_name <- paste0("3~", baseline_col)

        slope_class1_name <- paste0("1~", slope_col)
        slope_class2_name <- paste0("2~", slope_col)
        slope_class3_name <- paste0("3~", slope_col)

        # Create results row with all coefficients
        results_row <- data.table(
          imaging_feature = feature,
          modality = modality,

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

        # Add to results list (use parent scope assignment)
        results_list <<- c(results_list, list(results_row))

      }, error = function(e) {
        log_info(paste("Error fitting model for", feature, ":", e$message))
        failed_models <<- c(failed_models, paste0(modality, ":", feature))
      })
    }
  }

  # Combine all results into single data.table
  log_info("Combining results from all models...")

  if (length(results_list) > 0) {
    final_results <- rbindlist(results_list)

    # Save results to CSV
    output_path <- file.path(results_path, "multinomial_results.csv")
    fwrite(final_results, output_path)

    log_info(paste("Results saved to:", output_path))
    log_info(paste("Results shape:", nrow(final_results), "rows,",
                   ncol(final_results), "columns"))
  } else {
    log_info("Warning: No results to save - all models failed")
  }

  # Save failed models list
  if (length(failed_models) > 0) {
    failed_output_path <- file.path(results_path, "multinomial_failed.txt")
    writeLines(failed_models, failed_output_path)
    log_info(paste("Failed models list saved to:", failed_output_path))
  }

  # Summary statistics
  n_total <- total_features
  n_failed <- length(failed_models)
  n_converged <- n_total - n_failed

  log_info(strrep("=", 50))
  log_info("MULTINOMIAL REGRESSION SUMMARY")
  log_info(strrep("=", 50))
  log_info(paste("Total brain features:", n_total))
  log_info(paste("Successfully converged:", n_converged,
                 sprintf("(%.1f%%)", 100 * n_converged / n_total)))
  log_info(paste("Failed to converge:", n_failed,
                 sprintf("(%.1f%%)", 100 * n_failed / n_total)))

  if (n_failed > 0 && n_failed <= 10) {
    log_info(paste("Failed models:", paste(failed_models, collapse = ", ")))
  } else if (n_failed > 10) {
    log_info(paste("Failed models (first 10):",
                   paste(failed_models[1:10], collapse = ", ")))
    log_info(paste("... and", n_failed - 10, "more"))
  }

  log_info("Multinomial regression analysis completed.")
}
