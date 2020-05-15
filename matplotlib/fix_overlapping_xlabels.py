# Requires:
# Python, matplotlib
#
# Can be installed with:
# pip install matplotlib
#
# Description:
# Automatically increases the width of a figure until no more x-labels overlap.
# However, this will not work when using `plt.show()` but will function correctly
# when saving the figure.

import matplotlib.transforms as trans


def get_renderer(fig):
    """Get the renderer for a Matplotlib Figure.

    Args:
        fig (matplotlib.figure.Figure): [description]

    Returns:
        matplotlib.backend_bases.RendererBase: A Matplotlib renderer.
    """
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


def get_figure_bbox(fig, bbox) -> Tuple[float, float, float, float]:
    """Transforms a bounding box into units matching the dimension of `fig`.

    This can be used to transform pixel values to inches, which are internally
    used by Matplotlib.

    Args:
        fig (matplotlib.figure.Figure): A Matplotlib figure.
        bbox (matplotlib.transforms.Bbox): A bounding box.

    Returns:
        Tuple[float, float, float, float]: A bounding box in inches represented
            as a tuple `(x, y, width, height)`.
    """
    trans_bbox = trans.TransformedBbox(bbox, fig.transFigure.inverted())
    x = trans_bbox.xmin
    y = trans_bbox.ymin
    width = trans_bbox.width
    height = trans_bbox.height

    return (x, y, width, height)


def fix_overlapping_xlabels(fig, increment: float = 0.2):
    """Increases the width of `fig` until x-axis labels no longer overlap.

    Args:
        fig (matplotlib.figure.Figure): A Matplotlib figure.
        increment (float, optional): Inches by which `fig` will be incrementally
            widened. Large values can lead to unnecessarily wide figures.
            Defaults to 0.2.
    """
    renderer = get_renderer(fig)
    ax = fig.get_axes()[0]
    # Render once, otherwise it will run infinitely long
    ax.get_tightbbox(renderer)
    xlabel = ax.get_xticklabels()
    
    while True:
        previous_bbox = None
        finished = True
        for label in xlabel:
            bbox = label.get_window_extent(renderer)
            bbox = get_figure_bbox(fig, bbox)
            if previous_bbox:
                if previous_bbox[0] + previous_bbox[2] > bbox[0]:
                    figsize = fig.get_size_inches()
                    fig.set_size_inches(figsize[0]+increment, figsize[1])
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
