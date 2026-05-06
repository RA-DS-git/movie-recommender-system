from sklearn.metrics.pairwise import cosine_similarity


class ItemItemCF:
    def fit(self, ratings_df):
        self.train_df = ratings_df

        self.item_user_matrix = ratings_df.pivot(
            index="movieId",
            columns="userId",
            values="rating",
        ).fillna(0)

        self.sim_matrix = cosine_similarity(self.item_user_matrix)

        self.item_map = {
            mid: i for i, mid in enumerate(self.item_user_matrix.index)
        }

    def predict(self, user_profile, movie_id):
        if movie_id not in self.item_map:
            return self.train_df["rating"].mean()

        target_idx = self.item_map[movie_id]
        sim_scores = self.sim_matrix[target_idx]

        user_ratings = user_profile.set_index("movieId")["rating"]

        num, den = 0, 0

        for m_id, rating in user_ratings.items():
            if m_id in self.item_map:
                sim = sim_scores[self.item_map[m_id]]
                if sim > 0:
                    num += sim * rating
                    den += sim

        if den == 0:
            return self.train_df["rating"].mean()

        return num / den