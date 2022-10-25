import numpy as np
import pandas as pd
import pymoo.problems as pm
import plotly.graph_objects as go

from artifishswarm import AFSA
from artifishswarm.utils import PWrapper

@np.vectorize
def foo(func, a, b):
    return func(np.array([a, b]))

def main():
    # ackley = get_problem('ackley')
    # griewank = get_problem('griewank')
    # zakharov = get_problem('zakharov')
    # rastrigin = get_problem('rastrigin')
    # sphere = get_problem('sphere')
    opt_func = PWrapper(pm.get_problem('ackley'))

    afsa = AFSA(func=opt_func,
                dimensions=2,  # our function takes one value and returns one value
                population=50,  # other parameters chosen empirically
                max_iterations=100,
                vision=0.5,
                crowding_factor=0.9,
                step=0.1,
                search_retries=10,
                low=-2,
                high=2,
                save_history=True,
                rand_seed=12345  # the seed makes the results reproducible
                )

    afsa.run()
    res = afsa.fish
    # dfs = afsa.get_history()
    # dfs['z'] = [opt_func(np.array([row[0], row[1]])) for _, row in dfs.iterrows()]

    df = pd.DataFrame(res, columns=['x', 'y'])  # convert the result to dataframe, since its easier to feed into plotly
    fish_z = [opt_func(i) for i in res]  # add z values for all the fish
    df['z'] = fish_z

    x, y = np.mgrid[-2:2:0.01, -2:2:0.01]  # generates 2D coordinates mesh needed for 3D plot, not required for 2D plot
    z = foo(opt_func, x, y)  # map the function over mesh to get the 3rd dimension

    fig1 = go.Figure(data=[go.Surface(x=x, y=y, z=z, opacity=0.6)])  # the function that is being optimized

    # frames = []
    # for i in range(afsa.max_iterations):
    #     fdf = dfs.loc[dfs['epoch'] == i]
    #     frames.append(
    #         go.Frame(data=[go.Scatter3d(mode='markers', x=fdf[0], y=fdf[1], z=fdf['z'])],
    #                  layout_xaxis_range=[-3, 3],
    #                  layout_yaxis_range=[-3, 3])
    #     )

    fig2 = go.Figure(data=[go.Scatter3d(mode='text',
                                        text='🐠',  #  '🐟', '🐠'  if you don't like fish as markers, we won't be friends
                                        textfont={
                                            'size': 25,
                                            'color': 'yellow'
                                        },
                                        x=df['x'], y=df['y'], z=df['z'])])

    fig3 = go.Figure(fig1.data + fig2.data)  # combine the fish scatter plot with the optimized function plot
    fig3.update_layout(title='Fimshies', autosize=False,
                       width=1500, height=1000,
                       margin=dict(l=65, r=50, b=65, t=90))

    fig3.show()

    # fig = go.Figure(
    #     data=fig1.data,
    #     layout=go.Layout(
    #         title="Start Title",
    #         xaxis=dict(range=[-3, 3], autorange=False),
    #         yaxis=dict(range=[-3, 3], autorange=False),
    #         transition={'duration': 200, 'easing': 'linear', 'ordering': 'traces first'},
    #         updatemenus=[dict(
    #             type="buttons",
    #             buttons=[dict(label="Play",
    #                           method="animate",
    #                           args=[None])])]
    #     ),
    #     frames=frames
    # )
    #
    # fig.show()


if __name__ == '__main__':
    main()
