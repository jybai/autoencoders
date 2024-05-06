import argparse
import yaml
import os
import torch
from typing import List, Optional, Literal

from autoencoder_sweeper import AutoEncoderSweeper, AutoEncoderSweeperConfig
from buffer import CachedActivationsBuffer as ActivationsBuffer
from buffer import CachedActivationsBufferConfig as ActivationsBufferConfig
from utils import get_activation_size, layer_loc_to_act_site, Struct

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_cfg_yaml", type=str)
    argparser.add_argument("--acts_cache_dir", type=str)
    # argparser.add_argument("--model_save_dir", type=str)

    argparser.add_argument("--layer", type=int, nargs='+', default=[0])
    argparser.add_argument("--layer_loc", type=str, default="mlpout", 
                           choices=["residual", "mlp", "attn", "attn_concat", "mlpout"])

    argparser.add_argument("--parallelism", type=int, default=8)

    # sweepable hparams configs
    argparser.add_argument("--lr", type=float, nargs='+', default=[5e-5])
    argparser.add_argument("--beta1", type=float, nargs='+', default=[0.9])
    argparser.add_argument("--beta2", type=float, nargs='+', default=[0.999])
    argparser.add_argument("--lambda_reg", type=float, nargs='+', default=[5])
    argparser.add_argument("--warmup_percent", type=float, nargs='+', default=[0.3], help="Passed to pct_start in torch.optim.lr_scheduler.OneCycleLR.")
    
    # activation normalization configs
    argparser.add_argument("--act_norms", type=float, nargs='+', default=None, help="""The norms to use for layer activation renormalization. 
                           If set, then AutoEncoderMultiLayer will be used, otherwise AutoEncoder will be used""")
    argparser.add_argument("--act_renorm_type", type=List[Literal["linear", "sqrt", "log", "none"]], nargs='+', default=None,
                           help="""the type of renormalization to use for layer activations, one of "linear", "sqrt",
                                "log", "none". Activations are scaled by act_renorm_scale*(avg(norms)/norms[layer]), where norms are the
                                result of the act_renorm_type applied to act_norms. Only used if act_norms is set""")
    argparser.add_argument("--act_renorm_scale", type=float, nargs='+', default=None, help="""a global scale to apply to all 
                           activations after renormalization. Only used if act_norms is set""")

    argparser.add_argument("--expansion", type=int, nargs='+', default=[16])
    argparser.add_argument("--num_activations", type=int, default=int(2 ** 27), 
                           help="total number of tokens to train on, the dataset will wrap around as needed")
    argparser.add_argument("--model_batch_size", type=int, default=16)
    argparser.add_argument("--batch_size", type=int, default=2048)
    argparser.add_argument("--buffer_size", type=int, default=int(2 ** 20))

    argparser.add_argument("--steps_per_report", type=int, default=100)
    argparser.add_argument("--steps_per_save", type=int, default=10000)

    argparser.add_argument("--wb_project", type=str, default="SAE-Anthropic_setting")
    argparser.add_argument("--wb_entity", type=str, default="autoencoder666")
    argparser.add_argument("--wb_group", type=Optional[str], default=None)
    
    argparser.add_argument("--primary_device", type=str, default="cuda:0")
    argparser.add_argument("--offload_device", type=str, default="cpu")
    argparser.add_argument("--perform_offloading", action="store_true", help="""To be used when gpu memory is tight,
                           shuffles the encoder and buffer model back and forth from gpu to cpu to limit peak gpu memory usage""")

    argparser.add_argument("--refresh_progress", action="store_true")

    args = argparser.parse_args()

    with open(args.model_cfg_yaml) as f:
        model_cfg = yaml.safe_load(f)
        args.model_cfg = Struct(**model_cfg)

    args.n_dim = get_activation_size(args.model_cfg.model_name, args.layer_loc)
    print(f"infer act_size: {args.n_dim}")
    args.m_dim = [args.n_dim * r for r in args.expansion]
    args.total_steps = args.num_activations // args.batch_size
    args.act_site = layer_loc_to_act_site(args.layer_loc)
    print(f"infer act_site: {args.act_site}")

    return args

def main():
    args = parse_args()

    buffer_cfg = ActivationsBufferConfig(
        model_name=args.model_cfg.model_name,
        layers=args.layer,
        act_site=args.act_site,
        act_size=args.n_dim,
        dataset_name=args.model_cfg.dataset_name,
        dataset_split=args.model_cfg.dataset_split,
        buffer_size=args.buffer_size,
        device=args.primary_device,
        buffer_device=args.offload_device,
        offload_device=args.offload_device if args.perform_offloading else None,
        shuffle_buffer=True,
        model_batch_size=args.model_batch_size,
        samples_per_seq=None,
        max_seq_length=args.model_cfg.max_seq_length,
        cache_dir=args.acts_cache_dir,
        refresh_progress=args.refresh_progress,
    )
    buffer = ActivationsBuffer(buffer_cfg)

    # os.makedirs(args.model_save_dir, exist_ok=True)
    sweeper_cfg = AutoEncoderSweeperConfig(
        n_dim=args.n_dim,
        m_dim=args.m_dim,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        lambda_reg=args.lambda_reg,
        warmup_percent=args.warmup_percent,
        layer=args.layer,
        act_norms=args.act_norms,
        act_renorm_type=args.act_renorm_type,
        act_renorm_scale=args.act_renorm_scale,
        wb_project=args.wb_project,
        wb_entity=args.wb_entity,
        wb_group=args.wb_group,
        device=args.primary_device,
        total_activations=args.num_activations,
        batch_size=args.batch_size,
        parallelism=args.parallelism,
        steps_per_report=args.steps_per_report,
    )
    sweeper = AutoEncoderSweeper(sweeper_cfg, buffer)
    sweeper.run()

if __name__ == '__main__':
    main()

