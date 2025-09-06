from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_predictions(true_list, pred_list, test_lists=None):
    """
    Compute classification metrics: accuracy, macro precision, recall, F1.
    Assumes true_list and pred_list are lists of integers (label IDs).
    test_lists is optional and currently ignored.
    """
    acc = accuracy_score(true_list, pred_list)
    precision, recall, f1, _ = precision_recall_fscore_support(true_list, pred_list, average='macro')
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "macro_f1": f1
    }