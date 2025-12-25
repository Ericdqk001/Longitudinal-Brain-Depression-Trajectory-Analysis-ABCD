import logging
import shutil
import subprocess
from pathlib import Path

from modelling.scripts.mixed_effect import derive_individual_slopes
from preprocess.preprocess import preprocess


def fit_multinomial_models(
    data_store_path: Path,
    version_name: str = "dev",
    experiment_number: int = 1,
):
    """Call R script for multinomial logistic regression analysis."""
    logging.info("Starting multinomial regression analysis (R)...")

    # Get path to R script
    r_script_path = (
        Path(__file__).parent / "modelling" / "scripts" / "multinomial_function.R"
    )

    # Build R command
    # Source the R file and call the function
    r_command = (
        f"source('{r_script_path}'); "
        f"fit_multinomial_models("
        f"'{data_store_path}', "
        f"'{version_name}', "
        f"{experiment_number})"
    )

    cmd = ["Rscript", "-e", r_command]

    logging.info(f"Executing R script: {r_script_path}")
    logging.info("Note: First run may take time to install R packages...")

    # Run R script with real-time output streaming
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # Stream output in real-time
    for line in process.stdout:
        if line.strip():
            logging.info(f"[R] {line.rstrip()}")

    # Wait for process to complete
    process.wait()

    # Check return code
    if process.returncode != 0:
        logging.error(f"R script failed with return code {process.returncode}")
        raise RuntimeError(
            f"Multinomial regression R script failed with code {process.returncode}"
        )

    logging.info("Multinomial regression analysis completed successfully")


def main(
    data_store_path,
    version_name: str = "",
    experiment_number: int = 1,
):
    # Call the preprocess function from the prepare_img module
    preprocess(
        data_store_path=data_store_path,
        version_name=version_name,
        experiment_number=experiment_number,
    )

    # Call the derive_individual_slopes function from the mixed_effect module
    derive_individual_slopes(
        data_store_path=data_store_path,
        version_name=version_name,
        experiment_number=experiment_number,
    )

    # Call the fit_multinomial_models function (R script)
    fit_multinomial_models(
        data_store_path=data_store_path,
        version_name=version_name,
        experiment_number=experiment_number,
    )


if __name__ == "__main__":
    version_name = "dev"

    experiment_number = 2

    # Eddie staged data path
    data_store_path = Path("/exports/igmm/eddie/GenScotDepression")

    analysis_root_path = Path(
        data_store_path,
        "users",
        "eric",
        "depression_trajectories",
    )

    experiments_path = Path(
        analysis_root_path,
        version_name,
        "experiments",
    )

    if not experiments_path.exists():
        experiments_path.mkdir(parents=True, exist_ok=True)

    results_path = Path(
        experiments_path,
        f"exp_{experiment_number}",
    )

    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    local_log_file = Path("/tmp") / "experiment.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(local_log_file), logging.StreamHandler()],
    )

    main(
        data_store_path=data_store_path,
        version_name=version_name,
        experiment_number=experiment_number,
    )

    final_log_file = results_path / "experiment.log"

    try:
        shutil.move(str(local_log_file), str(final_log_file))
        print(f"Log file moved to: {final_log_file}")
    except Exception as e:
        print(f"Failed to move log file to mounted drive: {e}")
