import numpy as np
import pandas as pd
import plotly.express as px

from artifishswarm import AFSA
from artifishswarm.functions import ackley, beale, bukin6, eggholder, himmelblau, levi13, rastrigin, rosenbrock


@np.vectorize
def npvec_wrapper(func, a, b):
    return func(np.array([a, b]))


afsa_params = {
    'func': None,
    'dimensions': 2,  # all of our functions can work with 2 dimensions
    'population': 100,  # other parameters chosen empirically
    'max_iterations': 200,
    'vision': 0.02,
    'crowding_factor': 0.5,
    'step': 0.01,
    'search_retries': 10,
    'low': -2,
    'high': 2,
    'optimize_towards': 'min',
    'save_history': False,
}


functions = {
    'ackley': ackley,
    'beale': beale,
    'bukin6': bukin6,
    'eggholder': eggholder,
    'himmelblau': himmelblau,
    'levi13': levi13,
    'rastrigin': rastrigin,
    'rosenbrock': rosenbrock
}


def generate_plot(fish, func, range_min=-2, range_max=2, step=0.01, size_px=800):
        fish_array = fish

        fish_df = pd.DataFrame(fish_array, columns=['x', 'y'])  # convert the result to dataframe, since its easier to feed into plotly

        fish_z = [func(i) for i in fish_array]  # add z values for all the fish
        fish_df['z'] = fish_z

        x, y = np.mgrid[range_min:range_max:step, range_min:range_max:step]  # generates 2D coordinates mesh needed for 3D plot, not required for 2D plot
        z = npvec_wrapper(func, x, y)  # map the function over mesh to get the 3rd dimension

        fig1 = px.imshow(z, text_auto=True, x=np.arange(range_min, range_max, step), y=np.arange(range_min, range_max, step), color_continuous_scale=px.colors.sequential.Viridis)

        fig2 = px.scatter(x=fish_df['x'], y=fish_df['y'], color_discrete_sequence=['red'])
        fig2.update_xaxes(range=[range_min, range_max])
        fig2.update_yaxes(range=[range_min, range_max])

        fig1.add_trace(fig2.data[0])
        fig1.update_xaxes(range=[range_min, range_max])
        fig1.update_yaxes(range=[range_min, range_max])
        fig1.update_layout(
            autosize=False,
            width=size_px,
            height=size_px
        )

        return fig1


def main():
    for f_name, f in functions.items():
        afsa_params['func'] = f
        afsa = AFSA(**afsa_params)
        afsa.run()

        fig = generate_plot(afsa.fish, f)
        fig.write_image(f'images/{f_name}.png')


if __name__ == '__main__':
    main()