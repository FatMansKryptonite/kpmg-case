from ydata_profiling import ProfileReport
from seaborn import pairplot
import matplotlib.pyplot as plt


def make_profiles(data: dict, location: str = 'results/profiles') -> None:
    for key in data.keys():
        prof = ProfileReport(data[key])
        prof.to_file(output_file=f'{location}/{key}.html')


def make_pairwise_plot(data: dict,
                       group_by: str = 'missing',
                       key: str = 'data_train_fin',
                       location: str = 'results/pairplots') -> None:
    df = data[key]

    # Balance missing
    if not group_by == 'missing':
        g = df.groupby('missing')
        df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

    # Balance Classes
    g = df.groupby(group_by)
    df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))


    pairplot(df, hue=group_by)
    plt.savefig(f'{location}/{key}_{group_by}.pdf', bbox_inches='tight')
