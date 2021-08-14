## XAI module

### Module structure
- `data` - submodule with dataset loading and transformation
- `losses` - submodule with loss functions. The filter loss realized here. For details see: https://arxiv.org/abs/1710.00935
- `models` - all models are described here
- `modules` - building blocks for model
- `optim` - custom optimizers and LR schedulers
- `train` - training pipeline
- `tree` - the realization of tree building from cnn top filters. For details see: https://arxiv.org/abs/1802.00121
- `utils` - snapshot manipulations, logging, metrics writing to tensorboard 