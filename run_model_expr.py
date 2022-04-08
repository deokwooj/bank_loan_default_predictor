# Experiment script
"""  Bank’s loan applicants synthetic dataset build a ML model that predicts whether an applicant would default or pay back the loan."""
"""
Data Description:

This dataset  is loan data. When a customer applies for a loan, banks and other credit providers use models to determine 
whether to grant the loan based on the likelihood of the loan being repaid. You must implement a model 
that predicts loan repayment or default based on the data provided.

The dataset consists of synthetic data that is designed to exhibit similar characteristics to genuine loan data.
Explore the dataset, do the necessary data analysis, and implement a ML model to determine the best way to predict 
whether a loan applicant will fully repay or default on a loan.
"""

"""
Data Columns:

The dataset consists of the following fields:

• Loan ID: A unique Identifier for the loan information.

• Customer ID: A unique identifier for the customer. Customers may have more than one loan.

• Loan Status: A categorical variable indicating if the loan was paid back or defaulted. – Target variable

• Current Loan Amount: This is the loan amount that was either completely paid off, or the amount that was defaulted.

• Term: A categorical variable indicating if it is a short term or long term loan.

• Credit Score: A value between 0 and 800 indicating the riskiness of the borrowers credit history.

• Years in current job: A categorical variable indicating how many years the customer has been in their current job.

• Home Ownership: Categorical variable indicating home ownership. Values are "Rent", "Home Mortgage", and "Own". 
   If the value is OWN, then the customer is a home owner with no mortgage

• Annual Income: The customer's annual income

• Purpose: A description of the purpose of the loan.

• Monthly Debt: The customer's monthly payment for their existing loans

• Years of Credit History: The years since the first entry in the customer’s credit history • 
   Months since last delinquent: Months since the last loan delinquent payment

• Number of Open Accounts: The total number of open credit cards

• Number of Credit Problems: The number of credit problems in the customer records.

• Current Credit Balance: The current total debt for the customer

• Maximum Open Credit: The maximum credit limit for all credit sources.

• Bankruptcies: The number of bankruptcies

• Tax Liens: The number of tax liens.
"""
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


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

    return fig, axes


df = pd.read_csv('Loan Granting Binary Classification.csv', thousands=',')
df = df.dropna()

print(df.head(3))
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

if 0:
    print(len(df.columns))  # -> 19 cols
    print('-' * 30)
    print('c_ratio')
    print('-' * 30)
    for k, v in df.nunique().items():
        c_ratio = v / df.shape[0]
        print(f'{k}  -> {c_ratio:.4f}')
        if c_ratio < 1.e-4:
            cat_cols.append(k)
        else:
            num_cols.append(v)
    print('-' * 30)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

##################################################################
# Categorical Label encoding
##################################################################
trfm_dict = dict()
for col in cat_cols + res_cols:
    print(col)
    le = LabelEncoder()
    x = df[col].to_numpy().astype('str')
    x = le.fit_transform(x)
    df[col] = x
    trfm_dict[col] = le

if 0:
    fig, axes = plot_cols(df, res_cols, num_cols, 4, figname='num_cols', figsize=(10, 8))
    fig, axes = plot_cols(df, res_cols, cat_cols, 2, style='.', figname='num_cols', figsize=(10, 8))

##################################################################
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
##################################################################

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

##################################################################
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
preproc = Preprocessor(**preproc_kwargs)

preproc.fit(X_train, y_train)
print(X_train.shape)
X_train = preproc.transform(X_train)
print(X_train.shape)

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
emden.summary()
emden.plot_history()
emden.evaluate(X_test, y_test)
