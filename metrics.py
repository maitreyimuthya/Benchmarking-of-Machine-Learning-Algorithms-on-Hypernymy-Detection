from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
import matplotlib.pyplot as plt


def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def get_precision_recall_f1_score_support(y_true, y_pred):
    return precision_recall_fscore_support(y_true, y_pred, average='macro')


def plot_roc(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
