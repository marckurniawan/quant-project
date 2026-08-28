import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import classification_report

from src.models.labeling import create_labels
from src.features.engineering import generate_all_features  
from src.data.data_loader import load_and_validate_data


def prepare_training_data(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Prepare features and target labels for model training."""
    features = generate_all_features(df)
    labels = create_labels(df, threshold=threshold)

    dataset = features.join(labels[["target"]],how="inner")

    dataset = dataset.dropna()

    return dataset    


def f1_trading_signals(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculate macro F1 score for BUY and SELL signals, excluding HOLD."""
    return f1_score(
        y_true,
        y_pred,
        labels=[-1, 1],
        average="macro"
    )


def train_model(X: pd.DataFrame, y: pd.Series, random_state: int = 42, n_estimators: list[int] = [100, 200], max_depth: list[int] = [5, 10], n_splits: int = 5) ->  tuple[RandomForestClassifier, float, dict]:
    """Train and tune a Random Forest using time-series cross-validation."""
    
    model = RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state
        )

    param_grid = {
        "n_estimators": n_estimators,
        "max_depth": max_depth
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)

    trading_f1_scorer = make_scorer(f1_trading_signals)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring=trading_f1_scorer,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    return (
        grid_search.best_estimator_,
        grid_search.best_score_,
        grid_search.best_params_
    )


if __name__ == "__main__":
    df = load_and_validate_data("BBCA.JK", "2022-01-01", "2026-07-27")
    threshold = 0.01
    dataset = prepare_training_data(df, threshold=threshold)

    X = dataset.drop(columns=["target"])
    y = dataset["target"]
    
    print("Class distribution:")
    print(y.value_counts())
    print()
    print("Number of dataset rows:", len(dataset))
    model, validation_score, best_params = train_model(X, y)

    

    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
    print("Best cross-validated trading F1 score:", validation_score)
    print("Best parameters:", best_params)


