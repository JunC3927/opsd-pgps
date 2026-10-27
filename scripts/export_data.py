#!/usr/bin/env python3
"""Export the private PGPS data bundle. Does not train, evaluate, or upload."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geo_opsd.data import export_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--official-test", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True,
                        help="Historical training_images.json containing files/path/sha256 entries")
    parser.add_argument("--output", type=Path, required=True, help="New independent data-bundle directory")
    args = parser.parse_args()
    result = export_bundle(source_parquet=args.source_parquet, official_train=args.official_train,
                           official_test=args.official_test, image_manifest=args.image_manifest,
                           output=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
