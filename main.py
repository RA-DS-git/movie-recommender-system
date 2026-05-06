from src.data_loader import load_data
from src.preprocessing import build_movie_profiles
from src.content_based import ContentRecommender
from src.collaborative import ItemItemCF
from src.utils import train_test_split_users
from src.evaluation import evaluate_mae


def main():
    print("Loading data...")
    movies, tags, ratings = load_data()

    print("Building content-based model...")
    movie_profiles = build_movie_profiles(movies, tags)

    cb_model = ContentRecommender()
    cb_model.fit(movie_profiles)

    print("\nTop 5 similar movies to Toy Story:")
    print(cb_model.recommend(movie_id=1))

    print("\nTraining collaborative filtering model...")
    train_df, test_profiles, test_truths = train_test_split_users(
        ratings, test_user_ids=[1, 2, 3]
    )

    cf_model = ItemItemCF()
    cf_model.fit(train_df)

    mae = evaluate_mae(cf_model, test_profiles, test_truths)
    print(f"\nMAE: {mae:.4f}")


if __name__ == "__main__":
    main()