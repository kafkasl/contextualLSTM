#!/bin/sh

echo "#!/bin/sh
#SBATCH --job-name=make_wiki
#SBATCH --exclusive
#SBATCH -t30:59:00
#SBATCH --workdir=.
#SBATCH -o make_wiki-%J.out
#SBATCH -e make_wiki-%J.err
#SBATCH -N1
#SBATCH -n12

python -m gensim.scripts.make_wiki /gpfs/home/bsc19/bsc19277/contextualLSTM/data/enwiki-20170220-pages-articles.xml.bz2 /gpfs/home/bsc19/bsc19277/contextualLSTM/models/gensim" > job

sbatch < job
rm job
#SBATCH --dependency=afterany:753016

#SBATCH --mem=100000
