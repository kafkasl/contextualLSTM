#!/bin/sh

echo "#!/bin/sh
#SBATCH --job-name=word2ids
#SBATCH --exclusive
#SBATCH -t30:59:00
#SBATCH --workdir=.
#SBATCH -o word2ids-%J.out
#SBATCH -e word2ids-%J.err
#SBATCH -N1
#SBATCH -n12

export PYTHONPATH=$PYTHONPATH:/gpfs/home/bsc19/bsc19277/contextualLSTM/src

python /gpfs/home/bsc19/bsc19277/contextualLSTM/src/preprocess/words2ids.py -d /gpfs/home/bsc19/bsc19277/contextualLSTM/data/enwiki -w /gpfs/home/bsc19/bsc19277/contextualLSTM/models/eos/word2id_1000.pklz" > job

sbatch < job
rm job