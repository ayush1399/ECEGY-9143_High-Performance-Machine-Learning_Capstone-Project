# ECEGY-9143: High-Performance Machine Learning Capstone Project

> ## Setup
>
> **Create a new conda environment using the environment file in the repo**
>
> ```
> conda env create -f environment.yml
> ```
>
> ### Configuration
>
> ### Testing accuracy:
>
> To test accuracy choose your model (ResNet, ViT, Swin_V2, RPN, RPN_P, RPN_PQ, RPN_EE)
> Choose your dataset (Imagenet1KVal, ImagenetA, ImagenetC, ImagenetR, ImagenetV2)
>
> ```
> python eval.py --dataset ImagenetA --eval_mode acc --model ResNet --workers 4
> ```
>
> Note: by default you will get top-1 accuracy. To get top-5 accuracy simply pass the top5 flag as below:\_
>
> ```
> python eval.py --dataset ImagenetR --eval_mode acc --model ViT --workers 4 --top5
> ```

---

> ## Datasets
>
> ### Imagenet1K
>
> The imagenet 1K dataset is available at: [https://www.image-net.org/challenges/LSVRC/](https://www.image-net.org/challenges/LSVRC/). You will have to download all three (train, test and val) splits.
> _Note: You will have to sign up for an account and agree to their terms and conditions to access the dataset._
>
> ### Imagenet-A
>
> Imagenet-A is available here - [https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar)
>
> ### Imagenet-C
>
> Imagenet-C is available here - [https://zenodo.org/records/2235448](https://zenodo.org/records/2235448)
>
> ### Imagenet-R
>
> The imagenet-r dataset can be downloaded here - [https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
>
> ### Imagenet-V2
>
> The three Imagenet-v2 sets (matched-frequency, threshold and top-images) are available here - [https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main)
