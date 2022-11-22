# Basic example which shows how the afsa algorithm can be ran for simple quadratic function
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from artifishswarm import AFSA


def func(x):  # sample function to test the fishies behavior
    return -(x[0] ** 2)


def main():
    show_before = False
    afsa = AFSA(func=func,
                dimensions=1,  # our function takes one value and returns one value
                population=50,  # other parameters chosen empirically
                max_iterations=20,
                low=-2,
                high=3,
                vision=0.5,
                crowding_factor=0.98,
                step=0.5,
                search_retries=3,
                rand_seed=12345  # the seed makes the results reproducible
                )

    if show_before:
        plot(afsa.fish)

    afsa.run()
    plot(afsa.fish)


def plot(fish_array):
    x_data = np.arange(-2.5, 3.5, 0.1).reshape((-1, 1))
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
