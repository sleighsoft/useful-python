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
import queue
import time

from cairosvg import svg2png


class Consumer(multiprocessing.Process):

    def __init__(self, path_queue):
        super().__init__(daemon=True)
        self.path_queue = path_queue

    def run(self):
        proc_name = self.name
        try:
            while True:
                (inpath, outpath) = self.path_queue.get(timeout=1)
                print(inpath)
                with open(inpath, 'rb') as f:
                    svg = f.read()
                    svg2png(bytestring=svg, write_to=outpath)
                self.path_queue.task_done()
        except queue.Empty:
            print('Shutting down consumer')
            return


if __name__ == "__main__":
    s_time = time.time()
    parser = argparse.ArgumentParser(
        description=('Scans recursively for .svg files and converts them to '
                     '.png files. Will use all available CPU cores to process '
                     'files in parallel. Spawns number of CPU subprocesses '
                     'that get their tasks from a queue and exit on empty '
                     'queue.')
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

    paths = [p for p in inpath.rglob('*') if
             p.is_file() and p.suffix == '.svg']

    path_queue = multiprocessing.JoinableQueue()
    n_unfinished_paths = 0
    for p in paths:
        relative = p.relative_to(inpath)
        new_path = Path(
            str(outpath / relative.parent / relative.stem) + '.png')
        if not Path(new_path).exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            path_queue.put((str(p), str(new_path)))
            n_unfinished_paths += 1

    print(f'Unfinished paths: {n_unfinished_paths}')

    max_workers = multiprocessing.cpu_count()

    consumers = [Consumer(path_queue) for i in range(max_workers)]

    for w in consumers:
        w.start()

    path_queue.join()
    print(f'Finished {n_unfinished_paths} in {time.time()-s_time}s')
