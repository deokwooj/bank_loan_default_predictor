""" Ensemble model """
import numpy as np
from preprocessor import Preprocessor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch


class Ensemble:
    """ Ensemble model class for classifiers    """
    def __init__(self, prerpoc_kwargs, knn_kwargs, dt_kwargs, logi_kwargs, rforest_kwargs):
        self.preproc = Preprocessor(**prerpoc_kwargs)
        self.knn = KNN(**knn_kwargs)
        self.dt = DecisionTreeClassifier(**dt_kwargs)
        self.logi = LogisticRegression(**logi_kwargs)
        self.rforest = RandomForestClassifier(**rforest_kwargs)
        self.clfs = [self.knn, self.dt, self.logi, self.rforest]

    def fit(self, X, y):
        self.preproc.fit(X, y)
        X = self.preproc.transform(X)
        for clf_ in self.clfs:
            clf_.fit(X, y.flatten())

    def predict_proba(self, X):
        X = self.preproc.transform(X)
        pred = []
        for clf_ in self.clfs:
            pred.append(clf_.predict(X))
        pred_proba = np.array(pred).T.mean(axis=1)
        return pred_proba

    def predict(self, X):
        pred_vote = (self.predict_proba(X) > 0.5).astype(int)
        return pred_vote

    def evaluate(self, X, y):
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        conf_mat = confusion_matrix(y, y_pred)
        print('-' * 30)
        print(f'accuracy :{acc:.3f}, precision:{prec:.3f}, recall: {rec:.3f}, auc: {auc:.3f}')
        print('confusion matrix')
        print('-' * 30)
        print(conf_mat)
        print('-' * 30)
        self.eval_res = {'acc': acc, 'prec': prec, 'rec': rec, 'auc': auc, 'conf_mat': conf_mat}
