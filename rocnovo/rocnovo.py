"""Command-line entry point for CLIP and DeNovo training / prediction.

Usage examples:
    python -m rocnovo.rocnovo clip train --config clip.yaml --devices 0,1
    python -m rocnovo.rocnovo clip predict --config clip.yaml --checkpoint /path/clip.ckpt --output out_clip.csv --device 0
    python -m rocnovo.rocnovo denovo train --config denovo.yaml --devices 0
    python -m rocnovo.rocnovo denovo predict --config denovo.yaml --checkpoint /path/denovo.ckpt --output out_denovo.csv
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from rocnovo import run_clip, run_denovo


def _parse_devices(dev_str: Optional[str]) -> Optional[str]:
    """Helper to parse device string."""
    if dev_str is None:
        return None
    return dev_str


def main(argv: Optional[list[str]] = None) -> None:
    
    parser = argparse.ArgumentParser(prog="rocnovo", description="RocNovo command-line tool")
    mode_subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode: 'clip' or 'denovo'")

    # --- CLIP Mode ---
    p_clip = mode_subparsers.add_parser("clip", help="CLIP model operations (train/predict)")
    clip_subparsers = p_clip.add_subparsers(dest="command", required=True, help="Command: 'train' or 'predict'")

    # CLIP train
    p_clip_train = clip_subparsers.add_parser("train", help="Train CLIP model")
    p_clip_train.add_argument("--config", required=True, help="Path to CLIP config YAML file")
    p_clip_train.add_argument("--checkpoint", default=None, help="Model Checkpoints: Used for resuming training after an interruption.")
    p_clip_train.add_argument("--device", default=None, help="Comma-separated CUDA device ids, e.g. 0 or 0,1")

    # CLIP predict (newly added based on request)
    p_clip_predict = clip_subparsers.add_parser("predict", help="Run CLIP prediction and write CSV output")
    p_clip_predict.add_argument("--config", required=True, help="Path to CLIP config YAML file")
    p_clip_predict.add_argument("--checkpoint", required=True, help="CLIP model checkpoint path")
    p_clip_predict.add_argument("--output", required=True, help="Output CSV path")
    p_clip_predict.add_argument("--device", default="0", help="CUDA device id (single GPU), default 0")

    # --- DeNovo Mode ---
    p_denovo = mode_subparsers.add_parser("denovo", help="DeNovo model operations (train/predict)")
    denovo_subparsers = p_denovo.add_subparsers(dest="command", required=True, help="Command: 'train' or 'predict'")

    # DeNovo train
    p_denovo_train = denovo_subparsers.add_parser("train", help="Train DeNovo decoder")
    p_denovo_train.add_argument("--config", required=True, help="Path to DeNovo config YAML file")
    p_denovo_train.add_argument("--checkpoint", default=None, help="Optional path to save training checkpoints")
    p_denovo_train.add_argument("--device", default=None, help="Comma-separated CUDA device ids, e.g. 0 or 0,1")

    # DeNovo predict
    p_denovo_predict = denovo_subparsers.add_parser("predict", help="Run DeNovo prediction and write CSV output")
    p_denovo_predict.add_argument("--config", required=True, help="Path to DeNovo config YAML file")
    p_denovo_predict.add_argument("--checkpoint", required=True, help="DeNovo model checkpoint path")
    p_denovo_predict.add_argument("--output", required=True, help="Output CSV path")
    p_denovo_predict.add_argument("--device", default="0", help="CUDA device id (single GPU), default 0")

    # --- Parse args and execute ---
    args = parser.parse_args(argv)


    if args.mode == "clip":
        if args.command == "train":
            print(f"Running CLIP training with config: {args.config}")
            run_clip.train(args.config, args.checkpoint, _parse_devices(args.device))
        elif args.command == "predict":
            print(f"Running CLIP prediction with checkpoint: {args.checkpoint}")
            run_clip.predict(args.config,args.checkpoint, args.output, args.device)

    elif args.mode == "denovo":
        if args.command == "train":
            print(f"Running DeNovo training with config: {args.config}")
            run_denovo.train(args.config, args.checkpoint, _parse_devices(args.device))
        elif args.command == "predict":
            print(f"Running DeNovo prediction with checkpoint: {args.checkpoint}")
            run_denovo.predict(args.config, args.checkpoint, args.output, args.device)
    
    else:
            # This branch should be unreachable due to 'required=True'
            parser.print_help()

if __name__ == "__main__":
    main()