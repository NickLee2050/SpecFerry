#!/usr/bin/env python3
"""Format project C++ sources and headers with the repository clang-format style."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx', '.ipp', '.tpp'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='report formatting differences without modifying files')
    parser.add_argument('--clang-format', default='clang-format',
                        help='clang-format executable name or path')
    args = parser.parse_args()
    formatter = shutil.which(args.clang_format)
    if formatter is None:
        parser.error('clang-format was not found; activate SpecFerry and install dependencies '
                     'with: conda env update -n SpecFerry -f environment.yml')

    try:
        result = subprocess.run(
            ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
            cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f'cannot list project source files: {error}', file=sys.stderr)
        return 1

    paths = sorted({ROOT / os.fsdecode(name) for name in result.stdout.split(b'\0') if name})
    paths = [path for path in paths if path.suffix.lower() in EXTENSIONS
             and path.is_file() and not path.is_symlink()]
    if not paths:
        print('No C++ sources or headers found.')
        return 0

    options = ['--dry-run', '--Werror'] if args.check else ['-i']
    failed = False
    for path in paths:
        try:
            result = subprocess.run([formatter, '--style=file', *options, str(path)], cwd=ROOT)
        except OSError as error:
            print(f'cannot run clang-format: {error}', file=sys.stderr)
            return 1
        failed |= result.returncode != 0
    if failed:
        return 1
    action = 'Checked' if args.check else 'Formatted'
    print(f'{action} {len(paths)} C++ source/header files using .clang-format.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
