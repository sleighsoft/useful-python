# Requires
# Python 3.6+

import argparse
import concurrent.futures
import multiprocessing
import pathlib
from pathlib import Path
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Works similar to the linux rename utility. Supports '
                     'renaming files given a python regex.')
    )
    parser.add_argument(
        'inpath',
        type=str,
        help=('Directory to search for files.'))
    parser.add_argument(
        'pattern',
        type=str,
        help=('A regex pattern used to match files. May include '
              'Named Capture Groups: (?P<name>...)'))
    parser.add_argument(
        'replace',
        type=str,
        help=('The replacement string. :<name> or :<Number> will insert the '
              'corresponding capture group content. Unnamed capture groups '
              'start their numbering at 1. You have to also count the Named '
              ' Capture Groups before an unnamed one to determine its number.'))
    parser.add_argument(
        '-n', '-nono',
        action='store_true',
        help="No action: print names of files to be renamed, but don't rename."
    )
    parser.add_argument(
        '-r', '-recursive',
        action='store_true',
        help="Recursively search for matching files."
    )

    args = parser.parse_args()

    inpath = Path(args.inpath)

    search_pattern = re.compile(args.pattern)

    matches = []
    glob = '**/*' if args.r else '*'
    for p in inpath.glob(glob):
        match = search_pattern.search(str(p))
        if match:
            matches.append(match)

    group_pattern = re.compile(':<(.*?)>')
    groups_in_replace = group_pattern.findall(args.replace)

    for match in matches:
        path_str = match.group(0)
        replace_str = args.replace
        for group_name in groups_in_replace:
            if group_name in match.groupdict():
                replace_str = replace_str.replace(
                    f':<{group_name}>', match.group(group_name))
            elif int(group_name) < len(match.groups()) + len(match.groupdict()):
                replace_str = replace_str.replace(
                    f':<{group_name}>', match.group(int(group_name)))
        path = match.string.replace(path_str, replace_str)
        if args.n:
            print(f'Rename({match.string}, {path})')
        else:
            Path(match.string).rename(path)
