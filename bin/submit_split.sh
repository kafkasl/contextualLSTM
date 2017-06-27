#!/bin/sh
appName="splitr"
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

python /gpfs/home/bsc19/bsc19277/contextualLSTM/src/utils/split_1k.py -d /gpfs/home/bsc19/bsc19277/contextualLSTM/data/wikipedia/full_lists/train.list -o /gpfs/home/bsc19/bsc19277/contextualLSTM/data/wikipedia/train/

python /gpfs/home/bsc19/bsc19277/contextualLSTM/src/utils/split_1k.py -d /gpfs/home/bsc19/bsc19277/contextualLSTM/data/wikipedia/full_lists/test.list -o /gpfs/home/bsc19/bsc19277/contextualLSTM/data/wikipedia/test/

python /gpfs/home/bsc19/bsc19277/contextualLSTM/src/utils/split_1k.py -d /gpfs/home/bsc19/bsc19277/contextualLSTM/data/wikipedia/full_lists/valid.list -o /gpfs/home/bsc19/bsc19277/contextualLSTM/data/wikipedia/valid/
" > job

sbatch < job
rm job

