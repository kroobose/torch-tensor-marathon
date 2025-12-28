"""Main entry point for the Tensor Marathon CLI."""

import argparse
import sys

from torch_tensor_marathon.config import config
from torch_tensor_marathon.cli import TensorMarathonCLI


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PyTorch Tensor Shape Marathon - Practice tensor operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tensor-marathon                    # Start interactive mode (Japanese)
  tensor-marathon --lang en          # Start in English
  tensor-marathon --category einsum  # Jump to Einstein Summation problems
        """
    )

    parser.add_argument(
        "--lang",
        choices=["ja", "en"],
        default="ja",
        help="UI language (default: ja)",
    )

    parser.add_argument(
        "--category",
        help="Jump directly to a specific category",
        choices=[
            "reshape_permute",
            "indexing_slicing",
            "broadcasting",
            "gather_scatter",
            "einsum",
            "stacking_splitting",
            "advanced_ops",
            "dl_applications",
        ],
    )

    parser.add_argument(
        "--gemini-api-key",
        help="Gemini API key for dynamic problem generation",
    )

    args = parser.parse_args()

    # Set configuration
    config.set_language(args.lang)

    if args.gemini_api_key:
        config.set_gemini_api_key(args.gemini_api_key)

    # Start CLI
    try:
        cli = TensorMarathonCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
