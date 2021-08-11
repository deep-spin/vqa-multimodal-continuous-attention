# Multimodal Continuous Visual Attention Mechanisms
PyTorch implementation of the Deep Modular Co-Attention Networks (MCAN) with multimodal continuous attention. Follow this procedure to replicate the results reported in our paper [Multimodal Continuous Visual Attention Mechanisms](https://arxiv.org/abs/2104.03046).

## Requirements

We recommend to follow the procedure in the official [MCAN](https://github.com/MILVLG/mcan-vqa) repository in what concerns software and hardware requirements. We also use the same setup - see there how to organize the `datasets` folders. The only difference is that we don't use bottom-up features; instead you can download improved grid features from [here](https://github.com/facebookresearch/grid-feats-vqa) and place them in `./features/train2014`, `./features/val2014` and `./features/test2015`. 

## Training

To train a model with multimodal continuous attention, run this command:

```train
python3 run.py --RUN='train' --MAX_EPOCH=15 --M='mca' --SPLIT='train' --train_rnd='True' --n_iter=5 --VERSION=<VERSION>
```

This will load all the default hyperparameters. You can assign a name for you model by doing ```<VERSION>='name'```. You can add ```--SEED=87415123``` to reproduce the results reported in the paper.

## Evaluation

The evaluations of both the VQA 2.0 *test-dev* and *test-std* splits are run as follows:

```eval
python3 run.py --RUN='test' --CKPT_V=<VERSION> --CKPT_E=15 --M='mca' --n_iter=10 --plot_attention='True' --count='True'
```
and the result file is stored in ```results/result_test/result_run_<'PATH+random number' or 'VERSION+EPOCH'>.json```. The obtained result json file can be uploaded to [Eval AI](https://eval.ai/web/challenges/challenge-page/830/overview) to evaluate the scores on *test-dev* and *test-std* splits. ```--plot_attention='True'``` allows you to obtain the parameters of the continuous attention density (mixing coefficients, means, and covariance matrices of the mixture of Gaussians), useful for visualization purposes. ```--count='True'``` prints the number of examples for each K.

