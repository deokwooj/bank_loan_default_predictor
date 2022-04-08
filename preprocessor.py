""" Preprocessing parts """
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import f_regression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer


class Preprocessor:
    """ Preprocessor class to generate model input """
    def __init__(self, x_num_idx, use_feature_sel=True,
                 use_cat=True,
                 use_hot_enc=True,
                 var_threshold=1.e-3, kbest_p_val_threshold=1.e-3):
        self.x_num_idx = x_num_idx
        self.var_threshold = var_threshold
        self.kbest_p_val_threshold = kbest_p_val_threshold
        self.sc = StandardScaler()
        self.var_sel = VarianceThreshold(threshold=self.var_threshold)
        self.use_feature_sel = use_feature_sel
        self.num_kbest_sel = SelectKBest(f_regression, k='all')
        self.num_kbest_sel_idx_ = None
        self.use_cat = use_cat
        self.use_hot_enc = use_hot_enc
        if self.use_hot_enc:
            self.cat_he = OneHotEncoder(sparse=False)
        else:
            self.cat_he = None

    def get_x_num(self, X):
        return X[..., :self.x_num_idx]

    def get_x_cat(self, X):
        return X[..., self.x_num_idx:]

    def fit(self, X, y):
        X_num = self.get_x_num(X)
        self.sc.fit(X_num)
        X_num__ = self.sc.transform(X_num)
        assert all(np.abs(X_num__.mean(axis=0)) < 1.e-6)
        assert all(np.abs(1.0 - X_num__.var(axis=0)) < 1.e-6)

        if self.use_feature_sel:
            self.var_sel.fit(X_num)
            self.num_kbest_sel.fit(X_num, y.flatten())
            self.num_kbest_sel_idx_ = self.num_kbest_sel.pvalues_ < self.kbest_p_val_threshold

        if self.use_cat:
            X_cat = self.get_x_cat(X)
            if self.use_hot_enc:
                self.cat_he.fit(X_cat)

    def transform(self, X):
        X_num = self.get_x_num(X)
        X_num = self.sc.transform(X_num)
        if self.use_feature_sel:
            X_num = X_num[..., self.num_kbest_sel_idx_]

        if self.use_cat:
            X_cat = self.get_x_cat(X)
            if self.use_hot_enc:
                X_cat = self.cat_he.transform(X_cat)
            X_num_cat = np.hstack([X_num, X_cat])
            return X_num_cat
        else:
            return X_num
