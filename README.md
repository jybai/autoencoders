# A lightweight implementation of Sparse Autoencoder (SAE)

## Installation
```
pip -r requirements.txt
```

## Training
```
CUDA_VISIBLE_DEVICES="0" python train.py --model_cfg_yaml ./model_configs/Llama-2-7b-chat-hf.yaml \
  --layers 16 --layer_loc residual --lr 1e-3 --batch_size 8192 --model_batch_size 8 \
  --acts_cache_dir [ACTIVATION_CACHE_DIR] --model_save_dir [MODEL_CACHE_DIR]
```
The final model is saved in `MODEL_CACHE_DIR` and can be loaded with `AutoEncoder.load`.
The activations will be cached on disk in `ACTIVATION_CACHE_DIR`. 
During training, activations are loaded and pinned in CPU memory.
Refreshing the buffer would lazy-load activations from CPU to GPU memory.
I have find this to be a 2x speed up over the non-hierarchical-cache version.

Currently I only expose some configs from `ActivationsBufferConfig` and `AutoEncoderConfig` to the `train.py` manually.
A better way to implement it is to automatically add the arguments through `argparse` (future TODOs).
