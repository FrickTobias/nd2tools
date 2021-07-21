import re
from pathlib import Path
import sys
from collections import Counter

if sys.stderr.isatty():
    from tqdm import tqdm
else:
    def tqdm(iterable, **kwargs):
        return iterable


def is_1_2(s, t):
    """
    Determine whether s and t are identical except for a single character of
    which one of them is '1' and the other is '2'.
    """
    differences = 0
    one_two = {"1", "2"}
    for c1, c2 in zip(s, t):
        if c1 != c2:
            differences += 1
            if differences == 2:
                return False
            if {c1, c2} != one_two:
                return False
    return differences == 1


def guess_paired_path(path: Path):
    """
    Given the path to a file that contains the sequences for the first read in a
    pair, return the file that contains the sequences for the second read in a
    pair. Both files must have identical names, except that the first must have
    a '1' in its name, and the second must have a '2' at the same position.
    Return None if no second file was found or if there are too many candidates.
    >>> guess_paired_path(Path('file.1.fastq.gz'))  # doctest: +SKIP
    'file.2.fastq.gz'  # if that file exists
    """
    name = path.name
    # All lone 1 digits replaced with '?'
    name_with_globs = re.sub(r"(?<![0-9])1(?![0-9])", "?", name)
    paths = [p for p in path.parent.glob(name_with_globs) if is_1_2(str(p), str(path))]
    if len(paths) == 1:
        return paths[0]
    return None


#def get_iter_axes(nd2reader_parser_object, exclude):
#    all_axes_dict = nd2reader_parser_object.sizes.copy()
#    axes_dict = delete_keys(all_axes_dict, exclude)
#    axes_string = ''.join(axes_dict.keys())


class Summary(Counter):

    def print_stats(self, name=None, value_width=15, print_to=sys.stderr):
        """
        Prints stats in nice table with two column for the key and value pairs in
        summary
        :param name: name of script for header e.g. '__name__'
        :param value_width: width for values column in table
        :param print_to: Where to direct output. Default: stderr
        """
        # Get widths for formatting
        max_name_width = max(map(len, self.keys()), default=10)
        width = value_width + max_name_width + 1

        # Header
        print("=" * width, file=print_to)
        print(f"STATS SUMMARY - {name}", file=print_to)
        print("-" * width, file=print_to)

        # Print stats in columns
        for name, value in self.items():
            value_str = str(value)
            if type(value) is int:
                value_str = f"{value:>{value_width},}"
            elif type(value) is float:
                value_str = f"{value:>{value_width + 4},.3f}"

            print(f"{name:<{max_name_width}} {value_str}", file=print_to)
        print("=" * width, file=print_to)
