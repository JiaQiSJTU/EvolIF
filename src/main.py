# encoding = "utf-8"

from state import StateManager
import argparse
import json
import os
import random

from data_utils.utils import write_jsonl


def run(args):
    # Set random seed for reproducibility
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate multiple files based on the ID range
    for file_id in range(args.start_id, args.end_id + 1):
        # Load or initialize the state
        state_in_path = os.path.join(args.output_dir, f"state_{file_id}.json")
        if args.resume and os.path.exists(state_in_path):
            sm = StateManager.load_from_file(state_in_path)
        else:
            sm = StateManager()

        # Calculate the number of steps to run: total steps - steps already completed
        steps_to_run = max(0, args.steps - sm.cur_turn)

        # Run multiple steps
        for _ in range(steps_to_run):
            sm.step()

        # Write each step snapshot to JSONL
        snapshots_path = os.path.join(
            args.output_dir, f"snapshots_{file_id}.jsonl")
        write_jsonl(snapshots_path, sm.round_instruction_history)

        # Save the final state
        state_out_path = os.path.join(args.output_dir, f"state_{file_id}.json")
        sm.save_to_file(state_out_path)

        print(
            f"Generated files for ID {file_id}: {state_out_path}, {snapshots_path}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Generate multi-round instruction sequences and save state.")
    p.add_argument("--steps", type=int, default=80, help="Total number of steps (not incremental steps)")
    p.add_argument("--output-dir", type=str, default="./state", help="Output directory path")
    p.add_argument("--start-id", type=int, default=201, help="Starting ID")
    p.add_argument("--end-id", type=int, default=201, help="Ending ID")
    p.add_argument("--resume", action="store_true", help="Resume from existing state to continue generation")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
