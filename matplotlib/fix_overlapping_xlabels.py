# Requires:
# Python, matplotlib
#
# Can be installed with:
# pip install matplotlib
#
# Description:
# Automatically increases the width of a figure until no more x-labels overlap.

import matplotlib.transforms as trans


def get_renderer(fig):
    if fig._cachedRenderer:
        renderer = fig._cachedRenderer
    else:
        canvas = fig.canvas

        if canvas and hasattr(canvas, "get_renderer"):
            renderer = canvas.get_renderer()
        else:  # Some noninteractive backends have no renderer until draw time.
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            canvas = FigureCanvasAgg(fig)
            renderer = canvas.get_renderer()

    return renderer


def get_figure_bbox(fig, bbox):
    trans_bbox = trans.TransformedBbox(bbox, fig.transFigure.inverted())
    x = trans_bbox.xmin
    y = trans_bbox.ymin
    width = trans_bbox.width
    height = trans_bbox.height

    return (x, y, width, height)


def fix_overlapping_xlabels(fig, increment=0.2):
    renderer = get_renderer(fig)
    ax = fig.get_axes()[0]
    # Render once, otherwise it will run infinitely long
    ax.get_tightbbox(renderer)
    xlabel = ax.get_xticklabels()

    while True:
        previous_bbox = None
        finished = True
        for i, label in enumerate(xlabel):
            bbox = label.get_window_extent(renderer)
            bbox = get_figure_bbox(fig, bbox)
            if previous_bbox:
                if previous_bbox[0] + previous_bbox[2] > bbox[0]:
                    figsize = fig.get_size_inches()
                    fig.set_size_inches(figsize[0] + increment, figsize[1])
                    finished = False
                    break
            previous_bbox = bbox
        if finished:
            break


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    x = [i for i in range(50)]
    ax.plot(x)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    fix_overlapping_xlabels(fig)
    plt.show()
