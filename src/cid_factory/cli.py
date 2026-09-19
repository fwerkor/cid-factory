from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="cid-factory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--cid-repo")
    parser.add_argument("--cid-python")
    parser.add_argument("--state-dir")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    if args.cid_repo:
        os.environ["CID_FACTORY_CID_REPO"] = args.cid_repo
    if args.cid_python:
        os.environ["CID_FACTORY_CID_PYTHON"] = args.cid_python
    if args.state_dir:
        os.environ["CID_FACTORY_STATE_DIR"] = args.state_dir
    uvicorn.run(
        "cid_factory.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
