from ydata_profiling import ProfileReport
from seaborn import pairplot
import matplotlib.pyplot as plt
from utils import balance


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
        df = balance(df, replacement=False)

    # Balance Classes
    df = balance(df, on=group_by, replacement=False)

    pairplot(df, hue=group_by)
    plt.savefig(f'{location}/{key}_{group_by}.pdf', bbox_inches='tight')
