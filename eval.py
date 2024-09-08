""" Evaluation Helper function """
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score


def evaluate(preproc, clf, X_test, y_test):
    X_test = preproc.transform(X_test)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1_val= f1_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    print('-' * 30)
    print(f'accuracy :{acc:.3f}, precision:{prec:.3f}, recall: {rec:.3f}, f1_val:{f1_val:.3f}')
    print('confusion matrix')
    print('-' * 30)
    print(conf_mat)
    print('-' * 30)
    return acc, prec, rec, conf_mat
