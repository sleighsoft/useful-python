# Requires
# Python 3+, Pillow
#
# Can be installed with:
# pip install Pillow
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


def layout_list(l, cols):
    return [l[i : i + cols] for i in range(0, len(l), cols)]


def slice_col(l, col):
    return [l[i][col] for i in range(len(l))]


def load_image_if_exists(path):
    if os.path.exists(path):
        return Image.open(path)
    else:
        return None


def merge(output_file, images, cols, resize=-1, fill_color=0):
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

    merged_img.save(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("output_file", type=str, help="Name of the output " "file.")
    parser.add_argument(
        "-images",
        type=str,
        nargs="+",
        required=True,
        help="A list of image files. Order matters for final"
        " layouting. You can leave grid elements blank by specifying and empty path with ''.",
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

    args = parser.parse_args()

    merge(args.output_file, args.images, args.cols, args.resize, args.fill_color)
