import pandas as pd

try:
    from xgboost import XGBClassifier
except ImportError:  # xgboost is optional; fall back to scikit-learn
    XGBClassifier = None
    from sklearn.ensemble import HistGradientBoostingClassifier

FEATURE_COLUMNS = [
    "similarity",
    "evidence_available",
    "nli_entailment",
    "nli_contradiction",
    "nli_neutral",
    "percentage_difference",
    "percentage_difference_missing",
    "historical_change",
]


def train_risk_model():
    # This is intentionally a tiny demonstration training set. The project is
    # using a heuristic risk scorer for the live workflow, and the model should
    # be treated as experimental rather than a validated ESG classifier.
    training_data = pd.DataFrame(
        [
            [0.84, 1, 0, 1, 0, 35, 0, 20, 1],
            [0.80, 1, 1, 0, 0, 0, 0, 5, 0],
            [0.55, 0, 0, 0, 1, 0, 0, 10, 1],
            [0.75, 1, 0, 0, 1, 5, 0, 5, 0],
            [0.90, 1, 0, 1, 0, 25, 0, 15, 1],
            [0.65, 1, 1, 0, 0, 0, 0, 2, 0],
        ],
        columns=FEATURE_COLUMNS + ["risk_label"],
    )

    X_train = training_data[FEATURE_COLUMNS]
    y_train = training_data["risk_label"]

    if XGBClassifier is None:
        model = HistGradientBoostingClassifier(
            max_iter=10,
            max_depth=2,
            learning_rate=0.3,
            min_samples_leaf=1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model

    model = XGBClassifier(
        n_estimators=10,
        max_depth=2,
        learning_rate=0.3,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        min_child_weight=0,
        gamma=0,
        reg_lambda=0,
        reg_alpha=0,
        random_state=42,
        n_jobs=1,
        verbosity=1,
    )

    model.fit(X_train, y_train)
    return model
