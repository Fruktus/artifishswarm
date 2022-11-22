import math

import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from artifishswarm import AFSA


def func(x):
    return math.cos(2*x[0])


def main():
    crowding_factor = 1
    afsa = AFSA(func=func,
                dimensions=1,
                population=50,
                max_iterations=100,
                low=-3,
                high=3,
                vision=1,
                crowding_factor=crowding_factor,
                step=0.5,
                search_retries=3,
                rand_seed=12345
                )

    afsa.run()
    plot(afsa.fish)


def plot(fish_array):
    x_data = np.arange(-8, 8, 0.1).reshape((-1, 1))
    fig1 = px.line(
        x=x_data.reshape(-1),
        y=[func(x) for x in x_data],
    )
    fig2 = px.scatter(x=fish_array.reshape(-1), y=[func(x) for x in fish_array])
    fig2.update_traces(marker=dict(color='red'))
    fig3 = go.Figure(data=fig1.data + fig2.data,
                     layout_yaxis_range=[-3, 3])
    fig3.show()


if __name__ == '__main__':
    main()
