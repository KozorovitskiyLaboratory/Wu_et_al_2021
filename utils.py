
def simpleAxis(ax, displayX=True):
    """ function to remove the top, right, and conditionally bottom
        axes from an Axes/Subplot

    INPUTS:
        ax - an axes object
        displayX - whether to display the bottom axis"""

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_yaxis().tick_left()

    if displayX == 0:
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_ticks_position('none')
        ax.get_xaxis().set_ticklabels([])
    else:

        ax.get_xaxis().tick_bottom()
