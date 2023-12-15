#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=RPN_acc
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=5

module purge

singularity exec --nv \
                --overlay /scratch/km5939/hpml_project/overlay-15GB-500K.ext3:ro \
                /scratch/work/public/singularity/cuda10.2-cudnn7-devel-ubuntu18.04.sif\
                /bin/bash -c "source /ext3/env.sh; cd /scratch/km5939/Swin-Transformer; conda activate /scratch/km5939/.conda/hpml_env; cd ./kernels/window_process; python setup.py install; cd /vast/km5939/ECEGY-9143_High-Performance-Machine-Learning_Capstone-Project/; python eval.py --dataset ImagenetV2 --eval_mode acc --model RPN --workers 4 >> RPN_acc.txt; python eval.py --dataset ImagenetV2 --eval_mode acc --model RPN --workers 4 --top5 >> RPN_acc.txt"
