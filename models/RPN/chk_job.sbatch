#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=RPN_chk
#SBATCH --output=%x.out

module purge

singularity exec --nv \
		--overlay /scratch/km5939/swin_t/overlay-15GB-500K.ext3:ro \
		/scratch/work/public/singularity/cuda10.2-cudnn7-devel-ubuntu18.04.sif\
		/bin/bash -c "source /ext3/env.sh; cd /vast/km5939/co-advise-master; conda activate /scratch/km5939/.conda/swin; cd /scratch/km5939/Swin-Transformer/kernels/window_process; 
python setup.py install; cd /vast/km5939/co-advise-master; python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --resume ./ckpt.pth --eval  --data-path /vast/km5939/ImageNet"
