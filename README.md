# Data preparation
Download the ImageNet dataset in the webdataset format from [huggingface](https://huggingface.co/datasets/Cylarus/ImageNet).

# Environment setup
```
conda create -n bgpt python=3.8
conda activate bgpt
pip install accelerate==0.33.0 torchvision==0.19.1 webdataset omegaconf einops wandb
```

# Training
1. Download the pretrained tokenizer from [huggingface](https://huggingface.co/orres/btok_w_entropy_reg_0.77).
2. Config accelerate
3. Config the downloaded tokenizer path, dataset path and other training configurations in ```config/gpt_bin.yaml```
4. Launch training via
```
sh gpt_train.sh
```