#!/bin/sh
appName="lstm"
echo "#!/bin/sh
#SBATCH --job-name=$appName
#SBATCH --exclusive
#SBATCH -t30:59:00
#SBATCH --workdir=.
#SBATCH -o $appName-%J.out
#SBATCH -e $appName-%J.err
#SBATCH -N1
#SBATCH -n16
#SBATCH --gres gpu:4
#SBATCH --constraint=k80
#SBATCH --mem=100000

module purge && module load K80 cuda/8.0 mkl/2017.1 CUDNN/5.1.10-cuda_8.0 intel-opencl/2016 python/3.6.0+_ML

python /gpfs/home/bsc19/bsc19277/contextualLSTM/src/lstm/lstm.py \
        --data_path /gpfs/home/bsc19/bsc19277/contextualLSTM/data/wikipedia/full.list \
        --embeddings /gpfs/home/bsc19/bsc19277/contextualLSTM/models/eos/idWordVec_ \
        --model medium \
        --use_fp16 True \
        --word_to_id /gpfs/home/bsc19/bsc19277/contextualLSTM/models/eos/word2id_200.pklz" > job

sbatch < job
rm job

