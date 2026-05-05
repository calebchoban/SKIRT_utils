#!/bin/bash
#SBATCH -J {job_name}            # Job name
#SBATCH -o out.o            # Name of stdout output file
#SBATCH -e out.e            # Name of stderr error file
#SBATCH -p general          # Queue (partition) name
#SBATCH --nodes={nodes}           # Total # of nodes 
#SBATCH --ntasks-per-node={tasks_per_node} # Total number of mpi tasks per node
#SBATCH --cpus-per-task={cpus_per_task}   # number of cpus per mpi task = OMP_NUM_THREADS.
#SBATCH --mem={memory}           # Total RAM to be used by this job
#SBATCH -t {walltime}         # Run time (hh:mm:ss)
#SBATCH --mail-type=end,fail     # Send email at begin and end of job
#SBATCH -A r00380           # Project/Allocation name
#SBATCH --mail-user={email}   # Email to send all jobs alters to
#SBATCH -D .

# This is an example SLURM job scripts to run SKIRT on Big Red 200 at IU.
# There is a total of 128 cores per node so (ntasks-per-node) * (cpus-per-task) should always be less than or equal to 128.
# Max RAM is 256 GB.

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
date
module list

export SKI_NAME="run.ski"

# Print out the SLURM info
echo $SLURM_JOB_NUM_NODES
echo $SLURM_NTASKS_PER_NODE
echo $SLURM_CPUS_PER_TASK


# Run with (ntasks-per-node) tasks each with (cpus-per-task) cores. 
# skirt is a little odd in that you need to give the number of cores/threads using the -t argument to skirt and not to srun
# Below are a few helpful arguments to skirt:
# -m argument adds the current memory usage for each task log output
# -v argument makes each task log verbose outputting all the MPI chunk info
# -e argument runs SKIRT in emulation mode to check the max memory usage of source and medium system.
# -b batches output data in out.o. Full output available in log.txt file.
srun --ntasks-per-node=$SLURM_NTASKS_PER_NODE skirt -t $SLURM_CPUS_PER_TASK -m -b $SKI_NAME
date
exit
