# Requires
# Python 3.6+, cairo, cairosvg
#
# Can be installed with:
# conda install cairo
# pip install cairosvg

import argparse
import concurrent.futures
import multiprocessing
import pathlib
from pathlib import Path

from cairosvg import svg2png

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Scans recursively for .svg files and converts them to '
                     '.png files. Will use all available CPU cores to process '
                     'files in parallel. Spawns number of CPU subprocesses '
                     'that each receive a chunk of all files to process.')
    )
    parser.add_argument(
        'inpath',
        type=str,
        help=('Root directory containing .svg files to be converted. This will'
              ' recursively search for .svg files starting at `inpath`.'))
    parser.add_argument(
        'outpath',
        type=str,
        help=('Root directory where converted .svg files will be written to. '
              'This will recreate the directory structure of files found at '
              '`inpath`.'))

    args = parser.parse_args()

    inpath = Path(args.inpath)
    outpath = Path(args.outpath)

    print(f'Reading files from {inpath} and writing to {outpath}')

    paths = [p for p in inpath.rglob('*')
             if p.is_file() and p.suffix == '.svg']

    unfinished_paths = []
    for p in paths:
        relative = p.relative_to(inpath)
        new_path = Path(
            str(outpath / relative.parent / relative.stem) + '.png')
        if not Path(new_path).exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            unfinished_paths.append((p, new_path))

    def convert(id, paths):
        print(f'Starting process with id: {id} and paths: {len(paths)}')
        total_paths = len(paths)
        for i, (inpath, outpath) in enumerate(paths):
            with open(inpath, 'rb') as f:
                svg = f.read()
                svg2png(bytestring=svg, write_to=str(outpath))
            if i % max(1, int(len(paths) * 0.1)) == 0:
                print(f'{id}: {i}/{total_paths}')
        print(f'Finished {total_paths} files in process with id: {id}')

    print(f'Unfinished paths: {len(unfinished_paths)}')

    max_workers = multiprocessing.cpu_count()
    chunk_size = max(len(unfinished_paths) // max_workers, 1)
    chunks = []
    for i in range(max_workers):
        chunks.append(unfinished_paths[i * chunk_size:(i + 1) * chunk_size])

    chunks[max_workers - 1] = unfinished_paths[i * chunk_size:]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            futures.append(executor.submit(convert, i, chunk))

        for future in concurrent.futures.as_completed(futures):
            try:
                _ = future.result()
            except Exception as exc:
                print(exc)
