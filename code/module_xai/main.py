# code/xai/main.py
import argparse
from code.xai.api import run_xai


def parse_args():
    parser = argparse.ArgumentParser(description="XAI demo for orthopaedic rehab project")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to fusion input CSV"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="code/xai/demo_output",
        help="Directory to save XAI outputs"
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        default=None,
        help="Optional patient_id to explain"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_xai(
        input_path=args.input_path,
        out_dir=args.out_dir,
        patient_id=args.patient_id
    )
    print("\nXAI report:")
    print(report["explain_one_line"])


if __name__ == "__main__":
    main()