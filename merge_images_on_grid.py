# Requires
# Python 3+, Pillow, matplotlib, numpy
#
# Can be installed with:
# pip install Pillow matplotlib numpy
#
# Description:
# This tool accepts multiple images and a column count as input and will align
# these images on a grid with N-columns. The order the image files are passed
# to -images will determine the position in the grid. Grid is filled left to
# right, top to bottom.
#
# This is, for example, useful to merge multiple different matplotlib plots
# after they have been created. Similiar to using subplots in the first place.
#
# Example:
# ```
# python merge_images_on_grid.py -images image1.png image2.png -cols 1 -resize 0.5
#
# This will align the two images in two rows and resize them to 50% of their
# initial size.

import argparse
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def layout_list(l, cols):
    return [l[i : i + cols] for i in range(0, len(l), cols)]


def slice_col(l, col):
    return [l[i][col] for i in range(len(l))]


def load_image_if_exists(path):
    if os.path.exists(path):
        return Image.open(path)
    else:
        return None


def merge_and_plot(
    output_file,
    images,
    cols,
    resize=-1,
    fill_color=0,
    title=None,
    x_ticklabels=None,
    y_ticklabels=None,
    xlabel=None,
    ylabel=None,
    figsize=None,
    dpi=None,
):
    if figsize is not None:
        if isinstance(figsize, (float, int)):
            figsize = [s * figsize for s in plt.rcParams.get("figure.figsize").copy()]

    merged_img, row_heights, column_widths = merge(
        output_file, images, cols, resize, fill_color, save=False
    )

    row_heights.reverse()

    x_ticks = np.cumsum(column_widths) - np.array(column_widths) / 2
    y_ticks = np.cumsum(row_heights) - np.array(row_heights) / 2

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(merged_img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT), origin="lower")
    ax.set_title(title)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.set_yticklabels(reversed(y_ticklabels))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi)


def merge(output_file, images, cols, resize=-1, fill_color=0, save=True):
    assert len(images) % cols == 0, 'len("-image") % "-cols" != 0'

    images = list(map(load_image_if_exists, images))

    if resize > 0:
        for img in images:
            if img is None:
                continue
            width = img.width * resize
            height = img.width * resize
            img.thumbnail((width, height), Image.LANCZOS)

    images = layout_list(images, cols)

    column_widths = [0] * cols
    for row in images:
        for i, img in enumerate(row):
            width = img.width if img is not None else 0
            column_widths[i] = max(column_widths[i], width)

    row_heights = [0] * len(images)
    for i, row in enumerate(images):
        for img in row:
            height = img.height if img is not None else 0
            row_heights[i] = max(row_heights[i], height)

    total_width = sum(column_widths)
    total_height = sum(row_heights)
    merged_img = Image.new("RGB", (total_width, total_height), color=fill_color)

    x_offset = 0
    y_offset = 0
    for row_id, row in enumerate(images):
        x_offset = 0
        row_height = row_heights[row_id]
        for col_id, img in enumerate(row):
            column_width = column_widths[col_id]
            if img is None:
                x_offset += column_width
                continue
            width, height = img.size
            x = x_offset + (column_width // 2) - (width // 2)
            y = y_offset + (row_height // 2) - (height // 2)
            merged_img.paste(img, (x, y))
            x_offset += column_width
        y_offset += row_height

    if save:
        merged_img.save(output_file)

    return merged_img, row_heights, column_widths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("output_file", type=str, help="Name of the output " "file.")
    parser.add_argument(
        "-images",
        type=str,
        nargs="+",
        required=True,
        help="A list of image files. Order matters for final" " layouting.",
    )
    parser.add_argument(
        "-cols",
        type=int,
        required=True,
        help="Number of columns of the final layout. "
        "-cols must be able to evenly distribute len(-images)",
    )
    parser.add_argument(
        "-resize",
        type=float,
        default=-1.0,
        help="A ratio to resize images by. Will keep aspect "
        "ratio. Defaults to keep original size.",
    )
    parser.add_argument(
        "-fill_color",
        type=int,
        default=0,
        nargs="3",
        help="Default color to use for filling. Requires 3 values as RGB. Default is black.",
    )

    parser.add_argument(
        "-title",
        type=str,
        default=None,
        help="(Optional) Plot title. If set, will output a matplotlib plot instead of a plain image.",
    )
    parser.add_argument(
        "-x_ticklabels",
        type=str,
        nargs="+",
        default=None,
        help="(Optional) A list of x-tick labels. If set, will output a matplotlib plot instead of a plain image.",
    )
    parser.add_argument(
        "-y_ticklabels",
        type=str,
        nargs="+",
        default=None,
        help="(Optional) A list of y-tick labels. If set, will output a matplotlib plot instead of a plain image.",
    )
    parser.add_argument(
        "-xlabel",
        type=str,
        default=None,
        help="(Optional) x-axis label. If set, will output a matplotlib plot instead of a plain image.",
    )
    parser.add_argument(
        "-ylabel",
        type=str,
        default=None,
        help="(Optional) y-axis label. If set, will output a matplotlib plot instead of a plain image.",
    )
    parser.add_argument(
        "-figsize",
        type=float,
        nargs="+",
        default=None,
        help="(Optional) figsize of plot. Can be a scalar to scale figsize or a list of width, height. If set, will output a matplotlib plot instead of a plain image.",
    )
    parser.add_argument(
        "-dpi",
        type=int,
        default=None,
        help="(Optional) dpi of plot. If set, will output a matplotlib plot instead of a plain image.",
    )

    args = parser.parse_args()

    if (
        args.title is not None
        or args.x_ticklabels is not None
        or args.y_ticklabels is not None
        or args.xlabel is not None
        or args.ylabel is not None
        or args.figsize is not None
        or args.dpi is not None
    ):
        figsize = args.figsize
        if figsize is not None and len(figsize) == 1:
            figsize = figsize[0]
        merge_and_plot(
            args.output_file,
            args.images,
            args.cols,
            args.resize,
            args.fill_color,
            title=args.title,
            x_ticklabels=args.x_ticklabels,
            y_ticklabels=args.y_ticklabels,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            figsize=figsize,
            dpi=args.dpi,
        )
    else:
        merge(args.output_file, args.images, args.cols, args.resize, args.fill_color)

