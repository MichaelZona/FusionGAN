# Source Code of FGAN.

We write our code based on the great work of [DeblurGAN](https://github.com/KupynOrest/DeblurGAN)

## How to run

### Prerequisites
- NVIDIA GPU + CUDA
- Pytorch

## Train

If you want to train the model on your data run the following command to create image pairs:
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```
And then the following command to train the model

```bash
python ./DeblurGAN/train.py --dataroot /.path_to_your_data  --learn_residual --fineSize 256 --name your_expriment_name --checkpoints_dir /path_to_the_possible_checkpoint --display_id -1 --which_model_netG SAKStackSelfUResnetGenerator --dataset_mode context --continue_train
```
If you want to run ablation expriments, you can set --which_model_netG as:

- SAKStackSelfUResnetGeneratorWithoutS    (NoS)
- SAKStackSelfUResnetGeneratorWithoutK    (NoK)
- SAKStackSelfUResnetGeneratorWithoutStack (NoStack)



