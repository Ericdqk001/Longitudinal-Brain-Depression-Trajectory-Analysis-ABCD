#!/bin/bash
#$ -N analysis
#$ -cwd
#$ -o /gpfs/igmmfs/eddie/GenScotDepression/eric/jobs/
#$ -e /gpfs/igmmfs/eddie/GenScotDepression/eric/jobs/
#$ -l h_rt=24:00:00
#$ -l h_vmem=32G
#$ -pe sharedmem 8
#$ -M qdeng2@ed.ac.uk
#$ -m beas

echo "Starting longitudinal brain-depression trajectory analysis..."
echo "Waiting for stagein job to complete..."

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load required modules
echo "Loading miniconda..."
module load igmm/apps/miniconda3/24.9.2

# Initialize conda for bash
echo "Initializing conda..."
eval "$(conda shell.bash hook)"

# Activate conda environment
echo "Activating conda environment..."
conda activate /gpfs/igmmfs/eddie/GenScotDepression/eric/abcd_analysis

# Verify R is accessible through conda
echo "Verifying R installation..."
R --version | head -3

# Set Python path for imports
echo "Setting Python path for relative imports..."
export PYTHONPATH="/gpfs/igmmfs/eddie/GenScotDepression/eric/users/Eric/depression_trajectories/src:$PYTHONPATH"

# Navigate to source directory
cd /gpfs/igmmfs/eddie/GenScotDepression/eric/users/Eric/depression_trajectories/src

# Run the analysis
echo "Starting analysis with staged data..."
python main.py
