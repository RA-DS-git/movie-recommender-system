import pandas as pd


def load_data(data_path="data/"):
    movies = pd.read_csv(f"{data_path}/movies.csv")
    tags = pd.read_csv(f"{data_path}/tags.csv")
    ratings = pd.read_csv(f"{data_path}/ratings.csv")

    return movies, tags, ratings