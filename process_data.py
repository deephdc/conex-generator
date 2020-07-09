import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "run",
        type=str,
        help="run that should be processed"
        )
parser.add_argument(
        "--no-expand-depth",
        action="store_const",
        const=True,
        default=False,
        help="cut all values to the shortest depth"
        )
args = parser.parse_args()

expand_depth = not args.no_expand_depth
run = args.run

import src

src.data.clear_run(run)
src.data.raw.convert_run(run, expand_depth=expand_depth)
src.data.interim.align_run(run, expand_depth=expand_depth)
src.data.interim.clear_run(run)

