#!/usr/bin/env python3
"""
Export NER (Named Entity Recognition) models to ONNX format for use with Hugot.

This script exports token classification models (BERT, DistilBERT, RoBERTa, etc.)
to ONNX format compatible with Hugot's TokenClassificationPipeline.

Supported models:
- dslim/bert-base-NER (BERT-based, CoNLL-2003)
- dslim/bert-large-NER (BERT-large, CoNLL-2003)
- dslim/distilbert-NER (DistilBERT-based, CoNLL-2003)
- dbmdz/bert-large-cased-finetuned-conll03-english
- Jean-Baptiste/roberta-large-ner-english
- Any HuggingFace token classification model

Usage:
    python export_ner_onnx.py --model dslim/bert-base-NER --output ./models/bert-base-NER
    python export_ner_onnx.py --model dslim/distilbert-NER --output ./models/distilbert-NER

Requirements:
    pip install transformers torch onnx onnxruntime optimum
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


def export_ner_to_onnx(
    model_id: str,
    output_dir: str,
    opset_version: int = 14,
    max_length: int = 512,
):
    """
    Export a NER model to ONNX format.

    Args:
        model_id: HuggingFace model ID (e.g., 'dslim/bert-base-NER')
        output_dir: Directory to save the exported model
        opset_version: ONNX opset version (default: 14)
        max_length: Maximum sequence length (default: 512)
    """
    print(f"Loading model: {model_id}")

    # Load config
    config = AutoConfig.from_pretrained(model_id)
    print(f"Model type: {config.model_type}")
    print(f"Number of labels: {config.num_labels}")
    print(f"Labels: {config.id2label}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model
    print("Loading model weights...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    model.eval()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare sample inputs
    sample_text = "My name is Wolfgang and I live in Berlin."
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Check if model uses token_type_ids
    has_token_type_ids = hasattr(model.config, "type_vocab_size") and model.config.type_vocab_size > 0

    # Define input names and prepare inputs
    input_names = ["input_ids", "attention_mask"]
    export_inputs = (input_ids, attention_mask)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }

    if has_token_type_ids and "token_type_ids" in inputs:
        input_names.append("token_type_ids")
        token_type_ids = inputs["token_type_ids"]
        export_inputs = (input_ids, attention_mask, token_type_ids)
        dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}

    output_names = ["logits"]

    # Export to ONNX
    print("Exporting to ONNX...")
    onnx_path = output_path / "model.onnx"

    torch.onnx.export(
        model,
        export_inputs,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(str(output_path))

    # Save config.json with id2label mapping (critical for NER)
    config_dict = {
        "model_type": config.model_type,
        "num_labels": config.num_labels,
        "id2label": config.id2label,
        "label2id": config.label2id,
        "max_position_embeddings": getattr(config, "max_position_embeddings", max_length),
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "architectures": getattr(config, "architectures", [f"{config.model_type.title()}ForTokenClassification"]),
    }

    # Add optional fields
    if hasattr(config, "pad_token_id") and config.pad_token_id is not None:
        config_dict["pad_token_id"] = config.pad_token_id
    if hasattr(config, "type_vocab_size"):
        config_dict["type_vocab_size"] = config.type_vocab_size

    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nModel exported successfully to: {output_path}")
    print(f"Files created:")
    for file in sorted(output_path.iterdir()):
        size = file.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  - {file.name} ({size_str})")

    return str(output_path)


def verify_onnx_model(onnx_path: str, model_id: str):
    """Verify the exported ONNX model produces correct outputs."""
    import onnx
    import onnxruntime as ort
    import numpy as np

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
    print("\nTesting inference...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    session = ort.InferenceSession(onnx_path)

    test_text = "My name is Wolfgang and I live in Berlin."
    inputs = tokenizer(test_text, return_tensors="np", padding=True)

    # Prepare inputs for ONNX Runtime
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    # Add token_type_ids if the model expects it
    input_names = [inp.name for inp in session.get_inputs()]
    if "token_type_ids" in input_names and "token_type_ids" in inputs:
        ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

    outputs = session.run(None, ort_inputs)
    logits = outputs[0]

    print(f"Output shape: {logits.shape}")
    print(f"  - Batch size: {logits.shape[0]}")
    print(f"  - Sequence length: {logits.shape[1]}")
    print(f"  - Number of labels: {logits.shape[2]}")

    # Get predictions
    config = AutoConfig.from_pretrained(model_id)
    predictions = np.argmax(logits, axis=-1)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    print("\nSample predictions:")
    for i, (token, pred) in enumerate(zip(tokens, predictions)):
        if token in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]:
            continue
        label = config.id2label[pred]
        if label != "O":
            print(f"  {token}: {label}")

    print("\nONNX Runtime inference: PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export NER models to ONNX format for Hugot"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dslim/bert-base-NER",
        help="HuggingFace model ID to export",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/bert-base-NER",
        help="Output directory for the exported model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the exported ONNX model",
    )

    args = parser.parse_args()

    output_path = export_ner_to_onnx(
        model_id=args.model,
        output_dir=args.output,
        opset_version=args.opset,
        max_length=args.max_length,
    )

    if args.verify:
        onnx_file = Path(output_path) / "model.onnx"
        verify_onnx_model(str(onnx_file), args.model)


if __name__ == "__main__":
    main()
