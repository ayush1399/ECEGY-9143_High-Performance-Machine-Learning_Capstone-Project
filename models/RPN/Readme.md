## This sub-directory contains code for the RPN(Reduced Parameter Network) model

- The model consists of a Small DEIT NEtwork (Data Efficient Transformer) containing ~26-28 million Parameters, trained on Imagenet-1k using Cross-bias Inductive Distillation with Rednet as Involution Teacher and Resnet-based Convolution Teacher as based on the paper: [Co-advise: Cross Inductive Bias Distillation](https://openaccess.thecvf.com/content/CVPR2022/papers/Ren_Co-Advise_Cross_Inductive_Bias_Distillation_CVPR_2022_paper.pdf)

- We created an [chk_job.sbatch file](https://github.com/ayush1399/ECEGY-9143_High-Performance-Machine-Learning_Capstone-Project/blob/dev/models/RPN/chk_job.sbatch) for the model to run on NYU HPC cluster 

- Also, the model is run within an environment, and the configuration is provided along with the code.

