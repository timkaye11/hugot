#!/usr/bin/env python3
"""
Export Gemma 3 models to ONNX format for use with Hugot.

This script exports decoder-only Gemma 3 models to ONNX with KV-cache support
for efficient autoregressive generation.

Usage:
    python export_gemma3_onnx.py --model tiny-random/gemma-3 --output ./models/gemma3-tiny
    python export_gemma3_onnx.py --model onnx-community/gemma-3-1b-it-ONNX --output ./models/gemma3-1b --download-only

Requirements:
    pip install transformers torch onnx onnxruntime optimum
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def export_gemma3_to_onnx(
    model_id: str,
    output_dir: str,
    opset_version: int = 14,
    use_past: bool = True,
    num_kv_layers: int = None,
):
    """
    Export a Gemma 3 model to ONNX format with KV-cache support.

    Args:
        model_id: HuggingFace model ID (e.g., 'tiny-random/gemma-3')
        output_dir: Directory to save the exported model
        opset_version: ONNX opset version (default: 14)
        use_past: Whether to include past_key_values for KV-cache
        num_kv_layers: Number of KV cache layers (inferred from config if None)
    """
    print(f"Loading model: {model_id}")

    # Load config first to check model type
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"Model type: {config.model_type}")
    print(f"Config: num_hidden_layers={config.num_hidden_layers}, "
          f"num_key_value_heads={getattr(config, 'num_key_value_heads', 'N/A')}, "
          f"head_dim={getattr(config, 'head_dim', 'N/A')}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load model
    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare model configuration for Hugot
    num_hidden_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    vocab_size = config.vocab_size

    if num_kv_layers is None:
        num_kv_layers = num_hidden_layers

    print(f"Model dimensions: layers={num_hidden_layers}, kv_heads={num_kv_heads}, head_dim={head_dim}")

    # Prepare sample inputs
    batch_size = 1
    seq_length = 8

    sample_text = "Hello, how are you?"
    inputs = tokenizer(sample_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Adjust to fixed sequence length
    if input_ids.shape[1] < seq_length:
        pad_length = seq_length - input_ids.shape[1]
        input_ids = torch.cat([
            torch.full((batch_size, pad_length), tokenizer.pad_token_id or 0, dtype=torch.long),
            input_ids
        ], dim=1)
        attention_mask = torch.cat([
            torch.zeros((batch_size, pad_length), dtype=torch.long),
            attention_mask
        ], dim=1)

    position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)

    # Define input names
    input_names = ["input_ids", "attention_mask", "position_ids"]

    # Prepare dynamic axes
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
        "position_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }

    # Prepare inputs for export
    export_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    # Add past_key_values inputs if using KV-cache
    if use_past:
        cache_seq_length = 128  # Fixed cache size for Hugot
        past_key_values = []

        for i in range(num_kv_layers):
            key_name = f"past_key_values.{i}.key"
            value_name = f"past_key_values.{i}.value"

            input_names.extend([key_name, value_name])

            # Shape: (batch_size, num_kv_heads, cache_seq_length, head_dim)
            key_cache = torch.zeros(batch_size, num_kv_heads, cache_seq_length, head_dim)
            value_cache = torch.zeros(batch_size, num_kv_heads, cache_seq_length, head_dim)

            past_key_values.append((key_cache, value_cache))
            export_inputs[key_name] = key_cache
            export_inputs[value_name] = value_cache

            dynamic_axes[key_name] = {0: "batch_size", 2: "past_sequence_length"}
            dynamic_axes[value_name] = {0: "batch_size", 2: "past_sequence_length"}

        # Convert to tuple format expected by model
        export_inputs["past_key_values"] = tuple(past_key_values)
        export_inputs["use_cache"] = True

    # Define output names
    output_names = ["logits"]

    if use_past:
        for i in range(num_kv_layers):
            output_names.append(f"present.{i}.key")
            output_names.append(f"present.{i}.value")
            dynamic_axes[f"present.{i}.key"] = {0: "batch_size", 2: "total_sequence_length"}
            dynamic_axes[f"present.{i}.value"] = {0: "batch_size", 2: "total_sequence_length"}

    # Export to ONNX
    print("Exporting to ONNX...")
    onnx_path = output_path / "model.onnx"

    # Prepare forward arguments
    forward_args = (
        export_inputs["input_ids"],
        export_inputs["attention_mask"],
        export_inputs["position_ids"],
    )

    if use_past:
        forward_args = forward_args + (export_inputs["past_key_values"],)

    try:
        torch.onnx.export(
            model,
            forward_args,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
        )
    except Exception as e:
        print(f"Standard export failed: {e}")
        print("Trying optimum-based export...")
        try:
            from optimum.onnxruntime import ORTModelForCausalLM
            ort_model = ORTModelForCausalLM.from_pretrained(
                model_id,
                export=True,
                trust_remote_code=True,
            )
            ort_model.save_pretrained(str(output_path))
            print("Export via optimum succeeded!")
        except Exception as e2:
            print(f"Optimum export also failed: {e2}")
            raise

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(str(output_path))

    # Save config.json with Hugot-compatible fields
    hugot_config = {
        "model_type": config.model_type,
        "vocab_size": vocab_size,
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": getattr(config, "max_position_embeddings", 8192),
        "eos_token_id": config.eos_token_id if hasattr(config, "eos_token_id") else tokenizer.eos_token_id,
        "pad_token_id": config.pad_token_id if hasattr(config, "pad_token_id") else (tokenizer.pad_token_id or 0),
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
    }

    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(hugot_config, f, indent=2)

    print(f"Model exported successfully to: {output_path}")
    print(f"Files created:")
    for file in output_path.iterdir():
        print(f"  - {file.name}")

    return str(output_path)


def download_preconverted_onnx(model_id: str, output_dir: str):
    """
    Download a pre-converted ONNX model from HuggingFace.

    Args:
        model_id: HuggingFace model ID (e.g., 'onnx-community/gemma-3-1b-it-ONNX')
        output_dir: Directory to save the model
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading pre-converted ONNX model: {model_id}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=model_id,
        local_dir=str(output_path),
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )

    print(f"Model downloaded to: {output_path}")
    return str(output_path)


def verify_onnx_model(onnx_path: str):
    """Verify the exported ONNX model."""
    import onnx
    import onnxruntime as ort

    print(f"\nVerifying ONNX model: {onnx_path}")

    # Load and check model
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX model validation: PASSED")

    # Print input/output info
    print("\nModel inputs:")
    for inp in model.graph.input:
        shape = [dim.dim_value if dim.dim_value else dim.dim_param
                 for dim in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")

    print("\nModel outputs:")
    for out in model.graph.output:
        shape = [dim.dim_value if dim.dim_value else dim.dim_param
                 for dim in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")

    # Test inference
    print("\nTesting inference with ONNX Runtime...")
    session = ort.InferenceSession(onnx_path)

    print("ONNX Runtime session created successfully!")
    print(f"Available providers: {session.get_providers()}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export Gemma 3 models to ONNX format for Hugot"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiny-random/gemma-3",
        help="HuggingFace model ID to export",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/gemma3",
        help="Output directory for the exported model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Export without KV-cache support",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download pre-converted ONNX model (for onnx-community models)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the exported ONNX model",
    )

    args = parser.parse_args()

    if args.download_only:
        output_path = download_preconverted_onnx(args.model, args.output)
    else:
        output_path = export_gemma3_to_onnx(
            model_id=args.model,
            output_dir=args.output,
            opset_version=args.opset,
            use_past=not args.no_kv_cache,
        )

    if args.verify:
        onnx_file = Path(output_path) / "model.onnx"
        if onnx_file.exists():
            verify_onnx_model(str(onnx_file))
        else:
            # Look for ONNX files in subdirectories (onnx-community format)
            for onnx_file in Path(output_path).rglob("*.onnx"):
                print(f"\nFound ONNX file: {onnx_file}")
                verify_onnx_model(str(onnx_file))
                break


if __name__ == "__main__":
    main()
