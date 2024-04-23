from random import random

import torch
import torch.nn.functional as F
from transformer_lens.loading_from_pretrained import convert_hf_model_config

class Struct:
    """
    https://stackoverflow.com/questions/6866600/how-to-parse-read-a-yaml-file-into-a-python-object
    """
    def __init__(self, **entries): 
        self.__dict__.update(entries)

class TiedLinear(torch.nn.Module):
    """
    Tied linear layer for autoencoder

    For reference, see: https://github.com/openai/sparse_autoencoder/blob/8f74a1cbeb15a6a7e082c812ccc5055045256bb4/sparse_autoencoder/model.py#L87
    """

    def __init__(self, tied_to: torch.nn.Linear):
        super().__init__()
        self.tied_to = tied_to

    def forward(self, x):
        return F.linear(x, self.tied_to.weight.t(), bias=self.tied_to.bias)

    @property
    def weight(self):
        return self.tied_to.weight.t()

    @property
    def bias(self):
        return self.tied_to.bias


def optimizer_to(optim, device):
    """
    From https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
    """
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def truncate_seq(seq, max_length):
    offset = int(random() * (len(seq) - max_length))
    return seq[offset:offset + max_length]


def get_activation_size(model_name: str, layer_loc: str):
    assert layer_loc in [
        "residual",
        "mlp",
        "attn",
        "attn_concat",
        "mlpout",
    ], f"Layer location {layer_loc} not supported"
    model_cfg = convert_hf_model_config(model_name)

    if layer_loc == "residual":
        return model_cfg["d_model"]
    elif layer_loc == "mlp":
        return model_cfg["d_mlp"]
    elif layer_loc == "attn":
        return model_cfg["d_head"] * model_cfg["n_heads"]
    elif layer_loc == "mlpout":
        return model_cfg["d_model"]
    elif layer_loc == "attn_concat":
        return model_cfg["d_head"] * model_cfg["n_heads"]
    else:
        return None

def layer_loc_to_act_site(layer_loc):
    '''
    https://github.com/chepingt/sparse_dictionary/blob/717636e9870656811b307c308e860cbf4e585198/sae_utils/activation_dataset.py#L69
    '''
    assert layer_loc in [
        "residual",
        "mlp",
        "attn",
        "attn_concat",
        "mlpout",
    ], f"Layer location {layer_loc} not supported"

    if layer_loc == "residual":
        return "hook_resid_post"
    elif layer_loc == "attn_concat":
        return "attn.hook_z"
    elif layer_loc == "mlp":
        return "mlp.hook_post"
    elif layer_loc == "attn":
        return "hook_resid_post"
    elif layer_loc == "mlpout":
        return "hook_mlp_out"
    else:
        return None
