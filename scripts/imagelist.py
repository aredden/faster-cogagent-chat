from pathlib import Path
from argparse import ArgumentParser
from typing import Optional
from pydantic import BaseModel


class Args(BaseModel):
    input_path: Path
    recursive: bool = False
    output_file: Optional[Path]
    extensions: list[str] = ["jpeg", "jpg", "png", "webp"]


def get_args():
    arg_p = ArgumentParser(
        prog="ImageListMaker",
        description="Create a text list dataset of images and paths in a directory",
    )
    arg_p.add_argument(
        "-i",
        "--input-path",
        type=str,
        required=True,
    )
    arg_p.add_argument("-r", "--recursive", action="store_true")
    arg_p.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="Output text file, if not given, will default to [directory_name].images.txt",
    )
    arg_p.add_argument(
        "-e",
        "--extensions",
        nargs="+",
        default=["jpeg", "jpg", "png", "webp"],
        required=False,
    )
    return Args(**vars(arg_p.parse_args()))


args = get_args()

ip = args.input_path.resolve()
if args.recursive:
    paths = [
        p
        for p in ip.rglob("*")
        if p.is_file() and p.suffix[1:].lower() in args.extensions
    ]
else:
    paths = [
        p
        for p in ip.rglob("*")
        if p.is_file() and p.suffix[1:].lower() in args.extensions
    ]

numbered = "\n".join(f"{p.as_posix()} {ix}" for ix, p in enumerate(reversed(paths)))

if args.output_file:
    out_p = args.output_file
else:
    out_p = Path(f"{ip.stem}.images.txt")

out_p.write_text(numbered)
