from sklearn.metrics import mean_absolute_error


def evaluate_mae(model, test_profiles, test_truths):
    all_preds = []
    all_true = []

    for i in range(len(test_profiles)):
        truth = test_truths[i]

        preds = [
            model.predict(test_profiles[i], row["movieId"])
            for _, row in truth.iterrows()
        ]

        all_preds.extend(preds)
        all_true.extend(truth["rating"].tolist())

    return mean_absolute_error(all_true, all_preds)