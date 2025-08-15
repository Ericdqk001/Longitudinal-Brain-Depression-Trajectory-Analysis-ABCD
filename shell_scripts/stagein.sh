#!/bin/bash
#$ -N stagein
#$ -q staging
#$ -cwd
#$ -l h_rt=4:00:00
#$ -l h_vmem=8G
#$ -M qdeng2@ed.ac.uk
#$ -m beas

# Define source and destination paths
SOURCE=/exports/igmm/datastore/GenScotDepression/data/abcd/release5.1
DESTINATION=/exports/igmm/eddie/GenScotDepression/eric

echo "Starting data staging from datastore to Eddie filesystem..."
echo "Source: ${SOURCE}"
echo "Destination: ${DESTINATION}"

# Stage the ABCD release5.1 data
echo "Staging ABCD release5.1 data..."
rsync -rl ${SOURCE} ${DESTINATION}

# Verify staging completed successfully
if [ $? -eq 0 ]; then
    echo "Data staging completed successfully"
    echo "Staged data location: ${DESTINATION}/release5.1"

    # Show size of staged data
    du -sh ${DESTINATION}/release5.1

else
    echo "ERROR: Data staging failed!"
    exit 1
fi

echo "Ready for analysis phase"
