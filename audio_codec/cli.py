#!/usr/bin/env python3
import argparse
import importlib
import os
import time
import gc
import torch
from tqdm import tqdm
from .registry import CODEC_REGISTRY

def decoder_list():
    print("Available decoders:")
    for key, info in sorted(CODEC_REGISTRY.items(), key=lambda x: int(x[0])):
        print(f"  {key}. {info['name']}")

def gen_wav_paths(in_dir):
    """Yield every .wav path under in_dir (recursive)."""
    for root, _, files in os.walk(in_dir):
        for fn in files:
            if fn.lower().endswith(".wav"):
                yield os.path.join(root, fn)

def decode_folder(decoder_id, in_dir, out_dir, device_str):
    """
    Dynamically builds the decoder with any of:
      - hub_name
      - sample_rate
      - device
      - config & checkpoint (for SpeechTokenizer)
    then decodes all .wav under in_dir → out_dir with a tqdm bar.
    """
    if decoder_id not in CODEC_REGISTRY:
        print(f"Unknown decoder {decoder_id!r}. Run `list` to see valid IDs.")
        decoder_list()
        return

    info    = CODEC_REGISTRY[decoder_id]
    module  = importlib.import_module(info["module"])
    Decoder = getattr(module, info["class"])

    # Device selection
    can_cuda = torch.cuda.is_available()
    if device_str == "cuda" and not can_cuda:
        print("Warning: CUDA unavailable; falling back to CPU.")
    device = "cuda" if (device_str == "cuda" and can_cuda) else "cpu"

    # Build constructor kwargs dynamically
    ctor_kwargs = {"device": device}
    if "hub_name" in info:
        ctor_kwargs["hub_name"]    = info["hub_name"]
    if info.get("sample_rate") is not None:
        ctor_kwargs["sample_rate"] = info["sample_rate"]
    if "config" in info and "checkpoint" in info:
        ctor_kwargs["config_path"] = info["config"]
        ctor_kwargs["ckpt_path"]   = info["checkpoint"]

    print(f"\nLoading model {info['name']} with {ctor_kwargs}…")
    decoder = Decoder(**ctor_kwargs)

    try:
        os.makedirs(out_dir, exist_ok=True)

        # Count files (O(1) memory)
        total = sum(1 for _ in gen_wav_paths(in_dir))
        if total == 0:
            print("No WAV files found.")
            return

        print(f"\nDecoding {total} files with {info['name']} on {device}:\n")

        # Single-pass tqdm
        pbar = tqdm(total=total, unit="it", desc="Decoding", ncols=80)
        for src in gen_wav_paths(in_dir):
            start = time.perf_counter()
            decoder.decode_file(src, out_dir)
            elapsed = time.perf_counter() - start

            # cleanup intermediate memory
            torch.cuda.empty_cache()
            gc.collect()

            pbar.update(1)
            pbar.set_postfix_str(f"{elapsed:.2f}s")

        pbar.close()
        print("\nBatch decode complete.\n")

    finally:
        # Final cleanup
        del decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser(prog="audio-codec")
    sub    = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="Show available decoders")
    dec = sub.add_parser("decode", help="Decode a folder of WAVs")
    dec.add_argument("decoder_id", help="ID of decoder (see list)")
    dec.add_argument("in_dir",     help="Input folder path")
    dec.add_argument("out_dir",    help="Output folder path")
    dec.add_argument("device",     choices=["cpu","cuda"], help="cpu or cuda")

    args = parser.parse_args()
    if args.cmd == "list":
        decoder_list()
    else:
        decode_folder(
            decoder_id = args.decoder_id,
            in_dir      = args.in_dir,
            out_dir     = args.out_dir,
            device_str  = args.device
        )

if __name__ == "__main__":
    main()
