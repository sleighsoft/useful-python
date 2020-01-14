from itertools import takewhile, repeat


def count_lines(filename):
    """Fastest way to count the number of lines in a text file."""
    with open(filename, "rb") as f:
        bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
        return sum(buf.count(b"\n") for buf in bufgen)
