def train_test_split_users(ratings, test_user_ids):
    train_df = ratings[~ratings["userId"].isin(test_user_ids)]
    test_df = ratings[ratings["userId"].isin(test_user_ids)]

    test_profiles = []
    test_truths = []

    for uid in test_user_ids:
        user_data = test_df[test_df["userId"] == uid]

        profile = user_data.sample(frac=0.8, random_state=42)
        truth = user_data.drop(profile.index)

        test_profiles.append(profile)
        test_truths.append(truth)

    return train_df, test_profiles, test_truths