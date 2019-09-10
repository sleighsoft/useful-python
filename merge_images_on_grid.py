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
# This will align the two images in two rows and resize them to 50% of their inital
# size.

import argparse
from PIL import Image


def layout_list(l, cols):
    return [l[i:i+cols] for i in range(0, len(l), cols)]


def slice_col(l, col):
    return [l[i][col] for i in range(len(l))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('output_file', type=str, help='Name of the output '
                        'file.')
    parser.add_argument('-images', type=str, nargs='+', required=True,
                        help='A list of image files. Order matters for final'
                        ' layouting.')
    parser.add_argument('-cols', type=int, required=True,
                        help='Number of columns of the final layout. '
                        '-cols must be able to evenly distribute len(-images)')
    parser.add_argument('-resize', type=float, default=-1.0,
                        help='A ratio to resize images by. Will keep aspect '
                        'ratio. Defaults to keep original size.')

    args = parser.parse_args()

    assert len(args.images) % args.cols == 0, 'len("-image") % "-cols" != 0'

    images = list(map(Image.open, args.images))

    if args.resize > 0:
        for img in images:
            width = img.width * args.resize
            height = img.width * args.resize
            img.thumbnail((width, height), Image.LANCZOS)

    images = layout_list(images, args.cols)

    total_width = 0
    total_height = 0
    max_row_height = []
    max_col_width = []

    for row in images:
        total_width = max(total_width, sum([i.width for i in row]))
        max_row_height.append(max([i.height for i in row]))

    for col in range(args.cols):
        col = slice_col(images, col)
        total_height = max(total_height, sum([i.height for i in col]))
        max_col_width.append(max([i.width for i in col]))

    merged_img = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    y_offset = 0
    for row_id, row in enumerate(images):
        x_offset = 0
        row_height = max_row_height[row_id]
        for col_id, img in enumerate(row):
            col_width = max_col_width[col_id]
            width, height = img.size
            x = x_offset + (col_width//2) - (width//2)
            y = y_offset + (row_height//2) - (height//2)
            merged_img.paste(img, (x, y))
            x_offset += col_width
        y_offset += row_height

    merged_img.save(args.output_file)