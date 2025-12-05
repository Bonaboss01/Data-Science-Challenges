from sklearn.metrics import accuracy_score, classification_report

def evaluate_classification(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
