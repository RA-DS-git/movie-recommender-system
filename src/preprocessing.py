def build_movie_profiles(movies, tags):
    # Convert genres into tokens
    movies["genre_tokens"] = movies["genres"].str.replace("|", " ", regex=False)

    # Clean and deduplicate tags
    tags["tag"] = tags["tag"].str.lower()

    movie_tags = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(set(x)))
        .reset_index()
    )

    # Merge
    movie_profiles = movies.merge(movie_tags, on="movieId", how="left")

    # Combine into metadata text
    movie_profiles["metadata"] = (
        movie_profiles["genre_tokens"] + " " + movie_profiles["tag"].fillna("")
    )

    return movie_profiles