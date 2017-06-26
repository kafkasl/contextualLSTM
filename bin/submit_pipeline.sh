#!/bin/sh

echo "#!/bin/sh
#SBATCH --job-name=word2vec
#SBATCH --exclusive
#SBATCH -t30:59:00
#SBATCH --workdir=.
#SBATCH -o word2vec-%J.out
#SBATCH -e word2vec-%J.err
#SBATCH -N1
#SBATCH -n12

python /gpfs/home/bsc19/bsc19277/contextualLSTM/bin/preprocess.py -d /gpfs/home/bsc19/bsc19277/contextualLSTM/data/enwiki -s 500" > job

sbatch < job
rm job
#SBATCH --dependency=afterany:753016

#SBATCH --mem=100000
