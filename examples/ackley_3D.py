import numpy as np
import pandas as pd
import pymoo.problems as pm
import plotly.graph_objects as go

from artifishswarm import AFSA
from artifishswarm.utils import PWrapper


@np.vectorize
def npvec_wrapper(func, a, b):
    return func(np.array([a, b]))


def main():
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
    fish_array = afsa.fish

    fish_df = pd.DataFrame(fish_array, columns=['x', 'y'])  # convert the result to dataframe, since its easier to feed into plotly
    
    fish_z = [opt_func(i) for i in fish_array]  # add z values for all the fish
    fish_df['z'] = fish_z

    x, y = np.mgrid[-2:2:0.01, -2:2:0.01]  # generates 2D coordinates mesh needed for 3D plot, not required for 2D plot
    z = npvec_wrapper(opt_func, x, y)  # map the function over mesh to get the 3rd dimension

    problem_fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, opacity=0.6)])  # the function that is being optimized

    solution_fig = go.Figure(data=[go.Scatter3d(mode='text',
                                                text='🐠',  #  '🐟', '🐠'  if you don't like fish as markers, we won't be friends
                                                textfont={
                                                    'size': 25,
                                                    'color': 'yellow'
                                                },
                                                x=fish_df['x'], y=fish_df['y'], z=fish_df['z'])])

    complete_fig = go.Figure(problem_fig.data + solution_fig.data)  # combine the fish scatter plot with the optimized function plot
    complete_fig.update_layout(title='Fimshies', autosize=False,
                               width=1500, height=1000,
                               margin=dict(l=65, r=50, b=65, t=90))

    complete_fig.show()


if __name__ == '__main__':
    main()
