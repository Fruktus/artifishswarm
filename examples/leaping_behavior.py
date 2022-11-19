# Basic example which shows how the afsa algorithm can be ran for simple quadratic function
import math

import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from artifishswarm import AFSA


def func(x):
    return math.sin(x[0])


def main():
    leaping_enabled = True

    afsa = AFSA(func=func,
                dimensions=1,
                population=100,
                max_iterations=180,
                vision=0.5,
                crowding_factor=0.98,
                step=0.5,
                search_retries=3,
                rand_seed=5,
                leap_eps=0.00005 if leaping_enabled else 0,
                leap_scale=7,
                )

    afsa.run()
    fish_array = afsa.fish

    x_data = np.arange(-35, 35, 0.1).reshape((-1, 1))
    fig1 = px.line(
        x=x_data.reshape(-1),
        y=[func(x) for x in x_data],
    )
    fig2 = px.scatter(x=fish_array.reshape(-1), y=[func(x) for x in fish_array])
    fig2.update_traces(marker=dict(color='red'))

    fig3 = go.Figure(data=fig1.data + fig2.data)
    fig3.show()


if __name__ == '__main__':
    main()
