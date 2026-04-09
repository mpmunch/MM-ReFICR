#!/bin/bash

# --- Job Configuration ---
#SBATCH --job-name=reficr_training
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=60
#SBATCH --mem=96G

# --- Requeue Configuration ---
# Send SIGTERM 30 seconds before walltime limit
#SBATCH --signal=B:SIGTERM@30


# Set max number of times the job will be requeued
# 3 restarts = 4 total runs
max_restarts=3

# --- Requeue Logic ---
# Fetch the current restarts value from the job context
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*' | cut -d= -f2)
iteration=${restarts:-0} # If no restarts found, it's the first run (iteration 0)

# Dynamically set output and error filenames using job ID and iteration
# This puts logs in a 'slurm_logs' subdirectory (create it first!)
# E.g., slurm_logs/158647_0.out, slurm_logs/158647_1.out, etc.
log_dir="./slurm_logs"
mkdir -p "${log_dir}" # Create log directory if it doesn't exist
outfile="${log_dir}/${SLURM_JOB_ID}_${iteration}.out"
errfile="${log_dir}/${SLURM_JOB_ID}_${iteration}.err"
#SBATCH --output="${outfile}" # Redirect Slurm's main stdout
#SBATCH --error="${errfile}"  # Redirect Slurm's main stderr

echo "Starting iteration ${iteration} of job ${SLURM_JOB_ID}"
echo "Output file: ${outfile}"
echo "Error file: ${errfile}"

## Define a term-handler function executed on SIGTERM ##
term_handler()
{
    echo "Caught SIGTERM signal at $(date)"
    echo "Job ${SLURM_JOB_ID}, Iteration ${iteration}"



    if [[ $iteration -lt $max_restarts ]]; then
        echo "Requeuing job ${SLURM_JOB_ID} for iteration $((iteration + 1))"
        # Requeue the job, Slurm increments the restart count
        scontrol requeue ${SLURM_JOB_ID}
        # Exit cleanly after requeueing
        exit 0
    else
        echo "Maximum restarts (${max_restarts}) reached. Not requeueing."
        # Exit with an error code if max restarts are hit
        exit 1
    fi
}

# Trap SIGTERM to execute the term_handler
trap 'term_handler' SIGTERM

#######################################################################################
# --- Training Command ---

echo "Running singularity container..."

singularity exec --nv p9-reficr_latest.sif bash run_multi-gpu.sh


# If the script finishes successfully *before* the time limit, print a success message
echo "Job ${SLURM_JOB_ID} iteration ${iteration} completed successfully at $(date)."


exit 0