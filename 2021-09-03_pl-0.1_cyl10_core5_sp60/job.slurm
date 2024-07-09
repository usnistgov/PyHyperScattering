#!/bin/bash -l
#SBATCH --job-name=CYRSOXS      # Job name
#SBATCH --output=cyrsoxs.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=cyrsoxs.%j.out  # Stderr (%j expands to jobId)
#SBATCH --time=04:00:00         # walltime
#SBATCH --nodes=1               # Number of nodes requested (only one node on glados)
#SBATCH --ntasks=1              # Number of tasks(processes) (tasks distributed across nodes)
#SBATCH --ntasks-per-node=1     # Tasks per node
#SBATCH --cpus-per-task=1       # Threads per task (all cpus will be on same node)
#SBATCH --gres=gpu:turing:1     # number and type of GPUS to use (glados only has turing)
#SBATCH --gres-flags=enforce-binding

set -e

if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# load necessary modules
source ~/.typyEnv CLEAN
typyEnv --add cyrsoxs/0.9.0

# Check the morphology

python /home/pab2/membrane-model/checkH5.py *.h5

cat check.txt

echo "~~MORPHOLOGY CHECK RESULTS~~"
echo "~~~~~~~see images too~~~~~~~"
cat check.txt

echo "~~WITH MATERIALS~~"
#Store morphology data
ls -al Material*

## RUN YOUR PROGRAM ##
echo "RUNNING ON GPU"${CUDA_VISIBLE_DEVICES}
CyRSoXS_N3 *.h5

# Reduce the data

python /home/pab2/membrane-model/reduce.py *.h5

# assemble the artifacts

cp *_ISI.png /home/pab2/membrane-model/artifacts/ISI/
cp *raw280.png /home/pab2/membrane-model/artifacts/raw280/
cp *red280.png /home/pab2/membrane-model/artifacts/red280/
cp *IqE.png /home/pab2/membrane-model/artifacts/IqE/
cp *AR.png /home/pab2/membrane-model/artifacts/AR/
cp *.redp /home/pab2/membrane-model/artifacts/redp/
