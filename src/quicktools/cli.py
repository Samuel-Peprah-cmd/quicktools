"""Command-line interface for quicktools."""
import argparse
from quicktools import mathtools, strtools


def main() -> None:
    parser = argparse.ArgumentParser(prog="quicktools", description="Quick math and text utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_prime = subparsers.add_parser("is-prime", help="Check if a number is prime")
    p_prime.add_argument("number", type=int)

    p_stats = subparsers.add_parser("stats", help="Compute mean/median/stdev of numbers")
    p_stats.add_argument("numbers", type=float, nargs="+")

    p_slug = subparsers.add_parser("slugify", help="Convert text into a url-friendly slug")
    p_slug.add_argument("text")

    p_palin = subparsers.add_parser("is-palindrome", help="Check if text is a palindrome")
    p_palin.add_argument("text")

    args = parser.parse_args()

    if args.command == "is-prime":
        print(f"{args.number} is prime: {mathtools.is_prime(args.number)}")
    elif args.command == "stats":
        nums = args.numbers
        print(f"mean={mathtools.mean(nums):.4f} median={mathtools.median(nums):.4f} stdev={mathtools.std_dev(nums):.4f}")
    elif args.command == "slugify":
        print(strtools.slugify(args.text))
    elif args.command == "is-palindrome":
        print(strtools.is_palindrome(args.text))


if __name__ == "__main__":
    main()