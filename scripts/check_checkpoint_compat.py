import argparse
import os
import sys
from collections import Counter

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.model import RPNet  # noqa: E402


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def module_prefix(key):
    parts = key.split(".")
    if len(parts) >= 2 and parts[0] in {"stage1", "stage2", "stage3", "stage4", "stage5"}:
        return parts[0]
    return parts[0]


def print_items(title, items, limit=None):
    items = list(items)
    print(f"\n=== {title}: {len(items)} ===")
    if limit is None:
        limit = len(items)
    for item in items[:limit]:
        print(item)
    if len(items) > limit:
        print(f"... {len(items) - limit} more")


def print_prefix_summary(title, keys):
    counter = Counter(module_prefix(key) for key in keys)
    print(f"\n=== {title} by module ===")
    if not counter:
        print("(none)")
        return
    for name, count in counter.most_common():
        print(f"{name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Compare current RPNet structure with a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth.tar checkpoint.")
    parser.add_argument("--num-targets", type=int, default=4)
    parser.add_argument(
        "--trajectory-modules",
        action="store_true",
        help="Include legacy_current-only modules in the model surface.",
    )
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = strip_module_prefix(state_dict)

    model = RPNet(
        num_targets=args.num_targets,
        backbone_pretrained=False,
        enable_trajectory_modules=args.trajectory_modules,
    )
    model_state = model.state_dict()

    checkpoint_keys = set(state_dict.keys())
    model_keys = set(model_state.keys())

    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    common = sorted(model_keys & checkpoint_keys)
    shape_mismatch = [
        f"{key}: checkpoint {tuple(state_dict[key].shape)} vs model {tuple(model_state[key].shape)}"
        for key in common
        if tuple(state_dict[key].shape) != tuple(model_state[key].shape)
    ]

    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint tensors: {len(checkpoint_keys)}")
    print(f"model tensors: {len(model_keys)}")
    print(f"matched tensors: {len(common) - len(shape_mismatch)}")

    print_items("missing in checkpoint", missing, args.limit)
    print_prefix_summary("missing in checkpoint", missing)
    print_items("unexpected in checkpoint", unexpected, args.limit)
    print_prefix_summary("unexpected in checkpoint", unexpected)
    print_items("shape mismatch", shape_mismatch, args.limit)
    print_prefix_summary("shape mismatch", [item.split(":", 1)[0] for item in shape_mismatch])


if __name__ == "__main__":
    main()
