from sklearn.metrics import accuracy_score, classification_report, f1_score


def evaluate(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"[{model_name}] Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    return {"accuracy": accuracy, "f1": f1}
