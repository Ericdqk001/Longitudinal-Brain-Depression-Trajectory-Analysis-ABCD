import logging
import shutil
from pathlib import Path

from modelling.scripts.mixed_effect import derive_individual_slopes
from preprocess.preprocess import preprocess


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


if __name__ == "__main__":
    version_name = "dev"

    experiment_number = 1

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
