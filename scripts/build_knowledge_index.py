"""Build a source/model-identified handbook index; never use evaluation cases."""
import argparse
import json
from pathlib import Path

from fintech_support.knowledge import HandbookRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handbook", type=Path, default=Path("data/knowledge/handbook.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/knowledge/index.json"))
    parser.add_argument("--backend", choices=("hybrid", "lexical"), default="hybrid")
    parser.add_argument("--manifest", type=Path, default=Path("retrieval-manifest.json"))
    args = parser.parse_args()
    retriever = HandbookRetriever.from_path(args.handbook, backend=args.backend,
                                            model_manifest_path=args.manifest)
    retriever.save_index(args.output)
    print(json.dumps({"index_id": retriever.index_id, "identity": retriever.index_identity,
                      "chunks": len(retriever.chunks), "backend": retriever.backend,
                      "degraded": retriever.degraded, "output": str(args.output)}))


if __name__ == "__main__":
    main()
