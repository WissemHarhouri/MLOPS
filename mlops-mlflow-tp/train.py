import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Create / select experiment
mlflow.set_experiment("iris-mlops")

with mlflow.start_run():

    n_estimators = 400
    random_state = 42


    # Load data
    X, y = load_iris(return_X_y=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state
    )

    # Train
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="macro")

    # MLflow logging
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)
    print("Precision:", precision)
