"""Train a coreference resolution model with optional KG embedding fusion.

All CLI flags map 1‑to‑1 to the hydra/OMEGACONF overrides that are injected by
`launcher.py` in Azure ML. Keep flag names short and kebab‑cased while the
resulting attribute names remain snake_cased for convenience in Python.
"""

import argparse
from datetime import datetime


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments and return an ``argparse.Namespace``."""

    parser = argparse.ArgumentParser(
        description="Train a coreference resolution model with optional KG embedding fusion"
    )

    # Core model / schedule ---------------------------------------------------------------------------------
    parser.add_argument(
        "--model_name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Hugging Face model ID or local path of the base language model (required)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning‑rate for Adafactor (default: 3e‑5)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight‑decay for Adafactor (default: 0.01)",
    )

    # LR scheduler ------------------------------------------------------------------------------------------
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=1500,
        help="Number of warm‑up steps before linear decay (default: 500)",
    )
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=15000,
        help="Total optimisation steps to schedule (default: 5000)",
    )

    # Architecture tweaks -----------------------------------------------------------------------------------
    parser.add_argument(
        "--incremental_model_num_layers",
        type=int,
        default=1,
        help="Number of additional transformer layers on top of the LM encoder (default: 1)",
    )
    parser.add_argument(
        "--kg_fusion_strategy",
        type=str,
        choices=["baseline", "fusion", "add", "gating", "concat" ,"none"],
        default="baseline",
        help="How to combine text and KG context (default: baseline)",
    )
    parser.add_argument(
        "--kg_unknown_handling",
        type=str,
        choices=["zero_vector", "unk_embed"],
        default="unk_embed",
        help="How to handle unknown mentions (default: unk_embed)",
    )
    parser.add_argument(
        "--use_random_kg_all",
        action="store_true",
        help="Replace the entire KG context with random entities (default: False)",
    )
    parser.add_argument(
        "--use_random_kg_selective",
        action="store_true",
        help="Randomise only a subset of KG entities (default: False)",
    )

    # Optimisation misc -------------------------------------------------------------------------------------
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=2,
        help="Accumulate gradients over N mini‑batches (default: 2)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=120,
        help="Early‑stopping patience in validation steps (default: 120)",
    )

    # # Utility ----------------------------------------------------------------------------------------------
    # parser.add_argument(
    #     "--output-dir",
    #     type=str,
    #     default="./outputs",
    #     help="Where to write checkpoints and logs (default: ./outputs)",
    # )

    args = parser.parse_args()

    # Echo the resulting configuration with a timestamp for traceability
    print(f"[{datetime.utcnow().isoformat(timespec='seconds')}] Parsed arguments:\n{args}\n")

    return args


if __name__ == "__main__":
    parse_args()
