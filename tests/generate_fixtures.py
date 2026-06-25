from __future__ import annotations

import argparse
from pathlib import Path

from .fixtures import fixture_cases, write_fixture_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GPU compare fixture GDS pairs.")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for case in fixture_cases():
        old_path, new_path = write_fixture_pair(case, args.output_dir)
        print(f"{case.name}: {old_path} {new_path}")


if __name__ == "__main__":
    main()
