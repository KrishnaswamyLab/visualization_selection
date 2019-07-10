import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display


class TabWidget(object):

    def __init__(self):
        self.children = []
        self.titles = []
        self.entered = False

    def new_tab(self, title):
        tab = widgets.Output()
        self.children.append(tab)
        self.titles.append(title)
        return tab

    def display(self):
        self.widget = widgets.Tab(children=self.children)
        for i, title in enumerate(self.titles):
            self.widget.set_title(i, title)
        return display(self.widget)


def autorange(fig, scatter):
    xmin, xmax = np.min(scatter.x), np.max(scatter.x)
    ymin, ymax = np.min(scatter.y), np.max(scatter.y)
    xbuff = (xmax - xmin) * 0.1
    ybuff = (ymax - ymin) * 0.1
    fig.layout.xaxis.range = [xmin - xbuff, xmax + xbuff]
    fig.layout.yaxis.range = [ymin - ybuff, ymax + ybuff]


def slider(values, name, default=None):
    if default is None:
        default = values[0]
    return widgets.SelectionSlider(
        options=values,
        value=default,
        description=name,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True
    )


def plotly(data, label=None, c=None, contour=True):
    axis_layout = dict(
        autorange=False,
        showgrid=False,
        zeroline=False,
        showline=True,
        ticks='',
        showticklabels=False
    )
    size = np.repeat(10, data.shape[0])
    fig_data = dict(
        type='scattergl',
        x=data[:, 0],
        y=data[:, 1],
        mode='markers'
    )
    fig_data
    fig = go.FigureWidget(
        data=[fig_data],
        layout=go.Layout(
            xaxis=axis_layout,
            yaxis=axis_layout
        )
    )
    axis_layout = dict(
        autorange=False,
        showgrid=False,
        zeroline=False,
        showline=True,
        ticks='',
        showticklabels=False
    )
    fig = go.FigureWidget(
        data=[
            dict(
                type='scattergl',
                x=data[:, 0],
                y=data[:, 1],
                mode='markers',
            )
        ],
        layout=go.Layout(
            xaxis=axis_layout,
            yaxis=axis_layout,
            autosize=False,
            width=500,
            height=500,
        )
    )
    fig.layout.titlefont.size
    fig.layout.titlefont.size = 22
    fig.layout.titlefont.family = 'Rockwell'
    if label is not None:
        fig.layout.xaxis.title = label + '1'
        fig.layout.yaxis.title = label + '2'
    else:
        fig.layout.xaxis.title = ''
        fig.layout.yaxis.title = ''

    scatter = fig.data[0]
    scatter.marker.opacity = 0.7
    scatter.marker.size = size
    scatter.marker.color = c
    fig.layout.hovermode = 'closest'

    autorange(fig, scatter)

    if contour:
        contour = fig.add_histogram2dcontour(
            x=scatter.x, y=scatter.y, contours={'coloring': 'lines'},
            showscale=False, ncontours=20)
        contour.colorscale = [[i, "rgb({}, {}, {})".format(round(r * 255), round(g * 255), round(b * 255))]
                              for i, (r, g, b) in zip(
            np.linspace(0, 1, len(plt.cm.inferno.colors)),
            plt.cm.inferno.colors)]
        contour.reversescale = True
        contour.hoverinfo = 'skip'
        return fig, scatter, contour
    else:
        return fig, scatter


def to_hex(c, data):
    if c is None:
        c = np.repeat('#000000', data.shape[0])
    else:
        c -= np.min(c)
        c /= np.max(c)
        c = plt.cm.inferno(c)
        c = np.array([mpl.colors.to_hex(color) for color in c])
    return c


def parameter_plot(algorithm, results, truth, c=None):
    result = results[algorithm]
    Y_default = result['output'][result['default']]
    c = to_hex(c, Y_default)
    s = np.repeat(10, truth.shape[0])

    fig_true, scatter_true = plotly(truth, c=c, contour=False)
    fig, scatter, contour = plotly(Y_default, algorithm, c=c)

    def set_params(n_step=10, **kwargs):
        Y = result['output'][tuple(kwargs.values())]
        steps = 1 - np.linspace(0, 1, n_step)[::-1]**3
        x0 = scatter.x
        y0 = scatter.y
        for step in steps:
            with fig.batch_update():
                scatter.x = contour.x = (1 - step) * x0 + step * Y[:, 0]
                scatter.y = contour.y = (1 - step) * y0 + step * Y[:, 1]
                autorange(fig, scatter)

    widget_args = {}
    for param_name, params, value in zip(result['param_names'], result['params'], result['default']):
        widget_args[param_name] = slider(params, param_name, default=value)

    params_slider = widgets.interactive(set_params,
                                        **widget_args,
                                        n_step=widgets.fixed(50))

    for child in params_slider.children:
        child.layout.width = '290px'

    def hover_fn(trace, points, state):
        size = s.copy()
        color = c.copy()
        ind = points.point_inds[0]
        color[ind] = '#FF0000'
        size[ind] = 20
        scatter.marker.size = scatter_true.marker.size = size
        scatter.marker.color = scatter_true.marker.color = color

    scatter.on_hover(hover_fn)
    scatter_true.on_hover(hover_fn)

    dash = widgets.HBox([widgets.VBox([fig_true, params_slider]),
                         fig])
    display(dash)
