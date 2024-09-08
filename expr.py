import scipy
import matplotlib

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plot_cols(df, res_cols, cols, n_r, style='-', figname=None, figsize=None):
    n_c = int(np.ceil(len(cols) / n_r))
    fig, axes = plt.subplots(n_r, n_c, figsize=figsize)
    fig.suptitle(figname)
    for i, k in enumerate(cols):
        ax = axes[i % n_r][i // n_r]
        df[k].plot(ax=ax, style=style)
        df[res_cols].plot(ax=ax, secondary_y=True, style='.')
        ax.set_title(k, fontsize=8)
        ax.set_xticks([])

    plt.show(block=True)
    return fig, axes


df = pd.read_csv('Loan Granting Binary Classification.csv', thousands=',')
df = df.dropna()
print(df.head(5))
print(df.describe())
print(df.nunique())
print(df.dtypes)

assert not df.isnull().values.any()

res_cols = ['Loan Status']  # response output cols
cat_cols = ['Term', 'Years in current job', 'Home Ownership', 'Purpose']
num_cols = ['Current Loan Amount', 'Credit Score', 'Annual Income', 'Monthly Debt',
            'Years of Credit History', 'Number of Open Accounts', 'Number of Credit Problems',
            'Current Credit Balance', 'Maximum Open Credit', 'Bankruptcies', 'Tax Liens']
input_cols = cat_cols + num_cols

assert df['Loan ID'].nunique() == df['Customer ID'].nunique()
print(df.shape[0])

print('-' * 20)
print('categorical var')
print('-' * 20)
for col in cat_cols:
    print(col)
print('-' * 20)

print('-' * 20)
print('numerical var')
print('-' * 20)
for col in num_cols:
    print(col)
print('-' * 20)

breakpoint()

#################################################################
# Categorical Label encoding
##################################################################
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
trfm_dict = dict()
for col in cat_cols + res_cols:
    print(col)
    le = LabelEncoder()
    x = df[col].to_numpy().astype('str')
    x = le.fit_transform(x)
    df[col] = x
    trfm_dict[col] = le


fig, axes = plot_cols(df, res_cols, num_cols, 4, figname='num_cols', figsize=(10, 8))


fig, axes = plot_cols(df, res_cols, cat_cols, 2, style='.', figname='num_cols', figsize=(10, 8))


#################################################################
# Numpy data array generation
##################################################################
# input numerical x
x_num = df[num_cols].to_numpy()
# input categorical x
x_cat = df[cat_cols].to_numpy()
# output response y
# swap label 1 and 0 , now 1 indicate a default event
y = df[res_cols].to_numpy()
y = np.abs(1 - y)

assert x_num.shape[0] == x_cat.shape[0] and y.shape[0] == x_cat.shape[0]
default_rate = y.sum() / len(y)
num_loan_app = df['Loan ID'].nunique()
num_cstm_app = df['Customer ID'].nunique()
num_cases = df.shape[0]
print('=' * 100)
print(f'default_rate = {100 * default_rate:.2f}% out of num_cases = {num_cases}')
print(f'num_loan_app = {num_loan_app}, num_cstm_app = {num_cstm_app}')
print('=' * 100)
################################################################

##################################################################
# Train/Validation/Test data generation
##################################################################
# [x_num, x_cat, y]
# split data into train and test data
train_test_split = 0.1
from sklearn.model_selection import train_test_split

x_num_idx = x_num.shape[1]
X = np.hstack([x_num, x_cat])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)


#################################################################
# Model Build
##################################################################
# Start to build model from here
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

from preprocessor import Preprocessor
from sklearn.utils import Bunch

preproc_kwargs = Bunch(x_num_idx=x_num_idx, use_feature_sel=True,
                       use_cat=True, use_hot_enc=True,
                       var_threshold=1.e-3, kbest_p_val_threshold=1.e-3)

"""
preproc_kwargs = Bunch(x_num_idx=x_num_idx, use_feature_sel=False,
                       use_cat=False, use_hot_enc=False,
                       var_threshold=1.e-3, kbest_p_val_threshold=1.e-3)
"""
preproc = Preprocessor(**preproc_kwargs)
print(X_test.shape)
print(X_train.shape)
preproc.fit(X_train, y_train)
print(X_train.shape)
X_train = preproc.transform(X_train)



##################################################################
# KNN model fit
##################################################################
print('*'*50)
print('KNN')
print('*'*50)
knn_kwargs = Bunch(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=50, n_jobs=-1)
knn = KNN(**knn_kwargs)
knn.fit(X_train, y_train.flatten())

##################################################################
# KNN model Evaluation
##################################################################
from eval import evaluate

acc, prec, rec, conf_mat = evaluate(preproc, knn, X_test, y_test)

##################################################################
# DecisionTreeClassifier
##################################################################
from sklearn.tree import DecisionTreeClassifier
print('*'*50)
print('DecisionTreeClassifier')
print('*'*50)

dt_kwargs = Bunch(criterion='gini', splitter='best')
dt = DecisionTreeClassifier(**dt_kwargs)
dt.fit(X_train, y_train.flatten())
acc, prec, rec, conf_mat = evaluate(preproc, dt, X_test, y_test)

##################################################################
# Logistic Regression Model
##################################################################
from sklearn.linear_model import LogisticRegression
print('*'*50)
print('Logistic Regression Model')
print('*'*50)

logi_kwargs = Bunch(penalty='l1', solver='liblinear')
logi = LogisticRegression(**logi_kwargs)
logi.fit(X_train, y_train.flatten())
acc, prec, rec, conf_mat = evaluate(preproc, logi, X_test, y_test)


##################################################################
# RandomForestClassifier Model
##################################################################
from sklearn.ensemble import RandomForestClassifier
print('*'*50)
print('RandomForestClassifier')
print('*'*50)

rforest_kwargs = Bunch(n_estimators=200, criterion='entropy', max_depth=50, n_jobs=-1)
rforest = RandomForestClassifier(**rforest_kwargs)
rforest.fit(X_train, y_train.flatten())
acc, prec, rec, conf_mat = evaluate(preproc, rforest, X_test, y_test)


##################################################################
# Ensemble Model
##################################################################
from ensemble import Ensemble
print('*'*50)
print('Ensemble')
print('*'*50)

ens_clf = Ensemble(prerpoc_kwargs=preproc_kwargs,
                   knn_kwargs=knn_kwargs,
                   dt_kwargs=dt_kwargs,
                   logi_kwargs=logi_kwargs,
                   rforest_kwargs=rforest_kwargs)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

ens_clf.fit(X_train, y_train)
ens_clf.evaluate(X_test, y_test)


############################################################
# TensorFlow model
############################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
print('*'*50)
print('NN Embedding Dense Model')
print('*'*50)

preproc_kwargs = Bunch(x_num_idx=x_num_idx, use_feature_sel=False,
                       use_cat=True, use_hot_enc=False,
                       var_threshold=1.e-3, kbest_p_val_threshold=1.e-3)

from emb_dense import EmbDense
emden= EmbDense(prerpoc_kwargs = preproc_kwargs,dense_cells=[16, 8, 4])
emden.fit(X_train,y_train, epochs = 20, validation_split =0.2)

X, y = X_train,y_train
preproc = Preprocessor(**preproc_kwargs)
preproc.fit(X,y)
X = preproc.transform(X)
xx_cat = preproc.get_x_cat(X)
xx_num = preproc.get_x_num(X)
cat_sizes = xx_cat.max(axis=0).astype(int)
n_cat = xx_cat.shape[1]
n_num = xx_num.shape[1]
n_dim = X.shape[1]
print(f'n_cat->{n_cat}, n_num->{n_num}, n_dim->{n_dim}')
assert n_cat + n_num == n_dim
assert len(cat_sizes) == n_cat
import tensorflow as tf
x_in = tf.keras.layers.Input(shape=(n_dim,), name='model_input')
x_num = x_in[..., :x_num_idx]
x_cat = x_in[..., x_num_idx:]
dense_cells=[16, 8, 4]

cat_emb_out = []
for i, cat_size in enumerate(cat_sizes):
    x_cat_i = x_cat[..., i:i + 1]
    x_cat_i_out = tf.keras.layers.Embedding(input_dim=cat_size + 1,
                            output_dim=(cat_size + 1) // 2,
                            name=f'emb_{i}')(x_cat_i)
    x_cat_i_out = tf.keras.layers.Flatten()(x_cat_i_out)
    print(x_cat_i_out.shape)
    cat_emb_out.append(x_cat_i_out)

x_cat_emb_concat = tf.keras.layers.Concatenate(name='cat_emb_concat')(cat_emb_out)
x_num_cat_emb_concat = tf.keras.layers.Concatenate(name='num_cat_emb_concat')([x_num, x_cat_emb_concat])
x_dense = x_num_cat_emb_concat
for num_cell in dense_cells:
    x_dense = tf.keras.layers.Dense(units=num_cell, activation='relu')(x_dense)

x_out = tf.keras.layers.Dense(1, activation='sigmoid')(x_dense)
nn_clf  = tf.keras.models.Model(inputs=[x_in], outputs=[x_out])
nn_clf.summary()
nn_clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X = preproc.transform(X)
epochs = 100
validation_split =0.2
nn_clf.fit(X, y, epochs=epochs, validation_split=validation_split)
history = nn_clf.history.history

plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.legend()
plt.show(block=True)

xx= preproc.transform(X_test)
y_pred = nn_clf.predict(xx)
y_pred = (y_pred  > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)











