from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def fit(self, movie_profiles):
        self.movie_profiles = movie_profiles

        self.tfidf_matrix = self.vectorizer.fit_transform(
            movie_profiles["metadata"]
        )

        self.sim_matrix = cosine_similarity(self.tfidf_matrix)

        self.movie_to_idx = {
            mid: idx for idx, mid in enumerate(movie_profiles["movieId"])
        }

    def recommend(self, movie_id, top_n=5):
        idx = self.movie_to_idx[movie_id]

        scores = list(enumerate(self.sim_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

        return self.movie_profiles.iloc[
            [i[0] for i in scores]
        ][["title", "genres"]]