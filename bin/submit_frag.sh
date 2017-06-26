#!/bin/sh
appName="lstmGen"
echo "#!/bin/sh
#SBATCH --job-name=$appName
#SBATCH --exclusive
#SBATCH -t30:59:00
#SBATCH --workdir=.
#SBATCH -o $appName-%J.out
#SBATCH -e $appName-%J.err
#SBATCH -N1
#SBATCH -n16
#SBATCH --mem=100000

module purge && module load K80 cuda/8.0 mkl/2017.1 CUDNN/5.1.10-cuda_8.0 intel-opencl/2016 python/3.6.0+_ML
#module purge && module load K80 mkl/2017.0.098 cuda/7.5 CUDNN/5.1.3  intel-opencl/2016 python/2.7.12_ML
#export PYTHONPATH=$PYTHONPATH:/gpfs/home/bsc19/bsc19277/contextualLSTM/src
python /gpfs/home/bsc19/bsc19277/contextualLSTM/src/lstm/lstm_frag.py \
        --data_path /gpfs/home/bsc19/bsc19277/contextualLSTM/data/wikipedia/small.list \
        --embeddings /gpfs/home/bsc19/bsc19277/contextualLSTM/models/eos/idWordVec_ \
        --model small \
        --use_fp16 True \
        --word_to_id /gpfs/home/bsc19/bsc19277/contextualLSTM/models/eos/word2id_200.pklz" > job

sbatch < job
rm job
#SBATCH --dependency=afterany:753016
#SBATCH --gres gpu:0
#SBATCH --constraint=k80
