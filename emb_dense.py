""" Tensorflow model """
from matplotlib import pyplot as plt
from preprocessor import Preprocessor
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, roc_auc_score
from tensorflow import keras
from tensorflow.keras.layers import (Dense, Embedding, Concatenate, Flatten, Dropout)
from tensorflow.keras.models import Model


class EmbDense:
    """ Embedding + Dense model """

    def __init__(self, prerpoc_kwargs,
                 dense_cells=[16, 8, 4]):
        self.preproc = Preprocessor(**prerpoc_kwargs)
        self.dense_cells = dense_cells
        self.nn_clf = None

    @property
    def x_num_idx(self):
        return self.preproc.x_num_idx

    def build_model(self, X):
        X = self.preproc.transform(X)
        xx_cat = self.preproc.get_x_cat(X)
        xx_num = self.preproc.get_x_num(X)

        self.cat_sizes = xx_cat.max(axis=0).astype(int)
        self.n_cat = xx_cat.shape[1]
        self.n_num = xx_num.shape[1]
        self.n_dim = X.shape[1]
        print(f'n_cat->{self.n_cat}, n_num->{self.n_num}, n_dim->{self.n_dim}')
        assert self.n_cat + self.n_num == self.n_dim
        assert len(self.cat_sizes) == self.n_cat

        x_in = keras.layers.Input(shape=(self.n_dim), name='model_input')

        x_num = x_in[..., :self.x_num_idx]
        x_cat = x_in[..., self.x_num_idx:]
        cat_emb_out = []
        for i, cat_size in enumerate(self.cat_sizes):
            x_cat_i = x_cat[..., i:i + 1]
            x_cat_i_out = Embedding(input_dim=cat_size + 1,
                                    output_dim=(cat_size + 1) // 2,
                                    name=f'emb_{i}')(x_cat_i)
            x_cat_i_out = Flatten()(x_cat_i_out)
            print(x_cat_i_out.shape)
            cat_emb_out.append(x_cat_i_out)

        x_cat_emb_concat = Concatenate(name='cat_emb_concat')(cat_emb_out)
        x_num_cat_emb_concat = Concatenate(name='num_cat_emb_concat')([x_num, x_cat_emb_concat])
        x_dense = x_num_cat_emb_concat
        for num_cell in self.dense_cells:
            x_dense = Dense(units=num_cell, activation='relu')(x_dense)

        x_out = Dense(1, activation='sigmoid')(x_dense)
        nn_clf = Model(inputs=[x_in], outputs=[x_out])
        return nn_clf

    def fit(self, X, y, epochs=20, validation_split=0.2):
        self.preproc.fit(X, y)
        self.nn_clf = self.build_model(X)
        self.nn_clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        X = self.preproc.transform(X)
        self.nn_clf.fit(X, y, epochs=epochs, validation_split=validation_split)
        self.history = self.nn_clf.history.history

    def summary(self):
        self.nn_clf.summary()

    def predict_proba(self, X):
        X = self.preproc.transform(X)
        pred_proba = self.nn_clf.predict(X)
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

    def plot_history(self):
        plt.plot(self.history['loss'], label='loss')
        plt.plot(self.history['val_loss'], label='val_loss')
        plt.legend()
