from typing import Optional

import torch
import wandb
from autoencoder import *
# from buffer import ActivationsBuffer, ActivationsBufferConfig
from buffer import CachedActivationsBuffer as ActivationsBuffer
from buffer import CachedActivationsBufferConfig as ActivationsBufferConfig
import time
from tqdm.auto import tqdm, trange
from utils import *
import argparse
import gc
import yaml
import os

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_cfg_yaml", type=str)
    argparser.add_argument("--acts_cache_dir", type=str)
    argparser.add_argument("--model_save_dir", type=str)
    argparser.add_argument("--layers", type=int, nargs='+', default=[0])
    argparser.add_argument("--layer_loc", type=str, default="mlpout", 
                           choices=["residual", "mlp", "attn", "attn_concat", "mlpout"])

    argparser.add_argument("--lr", type=float, default=1e-4)
    argparser.add_argument("--beta1", type=float, default=0)
    argparser.add_argument("--beta2", type=float, default=0.999)
    argparser.add_argument("--num_activations", type=int, default=int(2e9), 
                           help="total number of tokens to train on, the dataset will wrap around as needed")
    argparser.add_argument("--model_batch_size", type=int, default=16)
    argparser.add_argument("--batch_size", type=int, default=8192)
    argparser.add_argument("--buffer_size", type=int, default=int(2 ** 20))
    argparser.add_argument("--lambda_reg", type=float, default=1e-3)

    argparser.add_argument("--steps_per_report", type=int, default=100)
    argparser.add_argument("--steps_per_save", type=int, default=10000)

    argparser.add_argument("--expansion", type=int, default=4)
    # argparser.add_argument("--n_dim", type=int, default=4096) # TODO: automatically infer n_dim given the selected layer from the model

    argparser.add_argument("--wb_name", type=Optional[str], default=None)
    argparser.add_argument("--wb_notes", type=Optional[str], default=None)
    argparser.add_argument("--wandb_project", type=str, default="autoencoder")
    argparser.add_argument("--wandb_entity", type=str, default="andrewbai")
    argparser.add_argument("--primary_device", type=str, default="cuda:0")
    argparser.add_argument("--offload_device", type=str, default="cpu")
    argparser.add_argument("--perform_offloading", action="store_true", help="To be used when gpu memory is tight, \
shuffles the encoder and buffer model back and forth from gpu to cpu to limit peak gpu memory usage")

    argparser.add_argument("--refresh_progress", action="store_true")

    args = argparser.parse_args()

    with open(args.model_cfg_yaml) as f:
        model_cfg = yaml.safe_load(f)
        args.model_cfg = Struct(**model_cfg)

    args.n_dim = get_activation_size(args.model_cfg.model_name, args.layer_loc)
    print(f"infer act_size: {args.n_dim}")
    args.m_dim = args.n_dim * args.expansion
    args.total_steps = args.num_activations // args.batch_size
    args.act_site = layer_loc_to_act_site(args.layer_loc)
    print(f"infer act_site: {args.act_site}")

    return args

def main():
    args = parse_args()

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wb_name, notes=args.wb_notes)

    buffer_cfg = ActivationsBufferConfig(
        model_name=args.model_cfg.model_name,
        layers=args.layers,
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

    os.makedirs(args.model_save_dir, exist_ok=True)
    encoder_cfg = AutoEncoderConfig(
        n_dim=args.n_dim,
        m_dim=args.m_dim,
        device=args.primary_device,
        lambda_reg=args.lambda_reg,
        tied=False,
        record_data=True,
        save_dir=args.model_save_dir,
    )
    encoder = AutoEncoder(encoder_cfg)

    wandb.config.update({
        "max_lr": args.lr,
        "num_activations": args.num_activations,
        "batch_size": args.batch_size,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "perform_offloading": args.perform_offloading,
        "encoder": encoder_cfg.__dict__,
        "buffer": buffer_cfg.__dict__,
    })

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), foreach=False)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.total_steps,
        pct_start=0.1
    )

    try:
        prev_time = time.time()
        for i in trange(args.total_steps):
            # If offloading is enabled and the buffer needs to be refreshed, offload the encoder and its optimizer to the
            # offload device to free up memory on the primary device, which is needed by the buffer to load the next batch
            # of activations.
            if args.perform_offloading and buffer.will_refresh(batch=args.batch_size):
                encoder = encoder.to(args.offload_device)
                optimizer_to(optimizer, args.offload_device)
                torch.cuda.empty_cache()
                acts = buffer.next(batch=args.batch_size).to(encoder_cfg.device, non_blocking=True)
                encoder = encoder.to(args.primary_device)
                optimizer_to(optimizer, args.primary_device)
                gc.collect()
            else:
                acts = buffer.next(batch=args.batch_size).to(encoder_cfg.device, non_blocking=True)

            # 0 in the second dimension since we are only using one layer
            enc, loss, l1, mse = encoder(acts[:, 0, :])
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if i % args.steps_per_report == 0 and i > 0:
                freqs, avg_fired, fvu = encoder.get_data()

                wandb.log({
                    "l1": l1.item(),
                    "mse": mse.item(),
                    "total_loss": loss.item(),
                    "ms_per_act": 1000 * (time.time() - prev_time) / (args.batch_size * args.steps_per_report),
                    "avg_neurons_fired": avg_fired,
                    "lr": scheduler.get_last_lr()[0],
                    "feature_density": wandb.Histogram(freqs.log10().nan_to_num(neginf=-10).cpu()),
                })

                if i % args.steps_per_save == 0:
                    encoder.save("chk")

                torch.cuda.empty_cache()
                prev_time = time.time()
    finally:
        # Save the model
        encoder.save("final")

if __name__ == '__main__':
    main()

