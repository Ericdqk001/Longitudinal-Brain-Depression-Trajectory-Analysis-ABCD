#!/bin/bash
#$ -N stageout
#$ -q staging
#$ -cwd
#$ -o /gpfs/igmmfs01/eddie/GenScotDepression/eric/jobs/
#$ -e /gpfs/igmmfs01/eddie/GenScotDepression/eric/jobs/
#$ -l h_rt=2:00:00
#$ -M qdeng2@ed.ac.uk
#$ -m beas

# Version name parameter (can be changed for different runs)
VERSION_NAME="dev"

echo "Starting stageout of analysis results..."
echo "Version: ${VERSION_NAME}"

# Source and destination paths
EDDIE_RESULTS_PATH="/gpfs/igmmfs01/eddie/GenScotDepression/eric/depression_trajectories/${VERSION_NAME}"
DATASTORE_BASE_PATH="/exports/igmm/datastore/GenScotDepression/users/Eric/depression_trajectories"

echo "Source: ${EDDIE_RESULTS_PATH}"
echo "Destination: ${DATASTORE_BASE_PATH}/${VERSION_NAME}"

# Check if source exists
if [ ! -d "$EDDIE_RESULTS_PATH" ]; then
    echo "ERROR: Source path does not exist: $EDDIE_RESULTS_PATH"
    exit 1
fi

# Create destination directory structure if it doesn't exist
mkdir -p "${DATASTORE_BASE_PATH}"

echo "Copying analysis results to DataStore..."

# Copy the entire version folder to DataStore
rsync -rlv --progress \
    "${EDDIE_RESULTS_PATH}/" \
    "${DATASTORE_BASE_PATH}/${VERSION_NAME}/"

# Check if copy was successful
if [ $? -eq 0 ]; then
    echo "Stageout completed successfully!"
    echo "Results saved to: ${DATASTORE_BASE_PATH}/${VERSION_NAME}/"

    # Show what was copied
    echo "Contents copied:"
    ls -la "${DATASTORE_BASE_PATH}/${VERSION_NAME}/"

    # Show size of copied data
    du -sh "${DATASTORE_BASE_PATH}/${VERSION_NAME}/"

    echo "Analysis results are now permanently stored on DataStore"
else
    echo "ERROR: Stageout failed!"
    exit 1
fi

echo "Stageout job completed"
