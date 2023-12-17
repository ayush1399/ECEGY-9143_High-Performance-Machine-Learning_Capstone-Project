#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=RPN_chk_v100
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=5
module purge

echo "Processor: $(lscpu | grep 'Model name' | awk -F ':' '{print $2}' | xargs)" >> RPNprune_perf_v.txt
echo "RAM: $(free -h | grep Mem: | awk '{print $4}')" >> RPNprune_perf_v.txt
echo "GPU: $(nvidia-smi -q | grep 'Product Name')" >> RPNprune_perf_v.txt
singularity exec --nv \
                --overlay /scratch/km5939/hpml_project/overlay-15GB-500K.ext3:ro \
                /scratch/work/public/singularity/cuda10.2-cudnn7-devel-ubuntu18.04.sif\
                /bin/bash -c "source /ext3/env.sh; cd /scratch/km5939/Swin-Transformer; conda activate /scratch/km5939/.conda/hpml_env; cd ./kernels/window_process; python setup.py install; cd /vast/km5939/ECEGY-9143_High-Performance-Machine-Learning_Capstone-Project/; python eval.py --dataset Imagenet1KTest --model RPN_P --eval_mode perf --batch_size 1 --workers 4 >> RPNprune_perf_v.txt; python eval.py --dataset Imagenet1KTest --model RPN_P --eval_mode perf --batch_size 4 --workers 4 >> RPNprune_perf_v.txt; python eval.py --dataset Imagenet1KTest --model RPN_P --eval_mode perf --batch_size 8 --workers 4 >> RPNprune_perf_v.txt; python eval.py --dataset Imagenet1KTest --model RPN_P --eval_mode perf --batch_size 16 --workers 4 >> RPNprune_perf_v.txt; python eval.py --dataset Imagenet1KTest --model RPN_P --eval_mode perf --batch_size 32 --workers 4 >> RPNprune_perf_v.txt"
