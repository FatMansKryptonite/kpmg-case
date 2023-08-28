from ydata_profiling import ProfileReport


def make_profiles(data: dict, location: str = 'profiles') -> None:
    for key in data.keys():
        prof = ProfileReport(data[key])
        prof.to_file(output_file=f'{location}/{key}.html')