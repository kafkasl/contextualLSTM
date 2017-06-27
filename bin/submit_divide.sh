#!/bin/sh
appName="divider"
echo "#!/bin/sh
#SBATCH --job-name=$appName
#SBATCH --exclusive
#SBATCH -t30:59:00
#SBATCH --workdir=.
#SBATCH -o $appName-%J.out
#SBATCH -e $appName-%J.err
#SBATCH -N1
#SBATCH -n12
#SBATCH --mem=100000

export PYTHONPATH="$PYTHONPATH:/gpfs/home/bsc19/bsc19277/contextualLSTM/src/"
python /gpfs/home/bsc19/bsc19277/contextualLSTM/bin/divide.py" > job

sbatch < job
rm job
