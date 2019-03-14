import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from functools import partial
import optuna
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

new_study = True

##

train_df = pd.read_csv('input/train.csv')
#train_df = pd.read_csv('input/train_min.csv')
test_df = pd.read_csv('input/test.csv')

do_lda = False
fix_data_skew = False

if fix_data_skew:
    trues = train_df.loc[train_df['target'] == 1]
    falses = train_df.loc[train_df['target'] != 1].sample(frac=1)[:len(trues)]
    train_df = pd.concat([trues, falses], ignore_index=True).sample(frac=1)
else:
    train_df = train_df
    
train_df.head()

X_test = test_df.drop('ID_code',axis=1)
X = train_df.drop(['ID_code','target'],axis=1)
y = train_df['target']

##

if do_lda:    
    lda = LDA(solver='svd', n_components=5, store_covariance=True)
    X_lda = pd.DataFrame(lda.fit_transform(X, y))
    X_test_lda = pd.DataFrame(lda.transform(X_test))
    X["lda"] = X_lda
    X_test["lda"] = X_test_lda

##

def objective(X, y, trial):
    p_loss = 'log'
    p_max_iter = trial.suggest_int('max_iter', 1000, 10000), #5000
    p_tol = trial.suggest_loguniform('tol', 1e-11, 1e-1), #1e-7
    p_alpha = trial.suggest_uniform('alpha', 0.001, 10), #0.3
    p_penalty = trial.suggest_categorical('penalty', ['none', 'l1', 'l2', 'elasticnet']), #(default)
    p_l1_ratio = trial.suggest_uniform('l1_ratio', 0.01, 0.99), #0.15

    fit_params = {
        
    }
    cl = SGDClassifier(n_jobs=4, loss=p_loss, max_iter=p_max_iter[0], tol=p_tol[0], alpha=p_alpha[0],
        penalty=p_penalty[0], l1_ratio=p_l1_ratio[0])
    score = cross_val_score(cl, X, y, scoring='roc_auc', cv=5, fit_params=fit_params)
    score = np.mean(score)
    print(score)
    return 1.0 - score

def main():
    f = partial(objective, X, y)
    if new_study:
        study = optuna.create_study(study_name='log_reg_study', storage='sqlite:///storage.db')
    else:
        study = optuna.Study(study_name='log_reg_study', storage='sqlite:///storage.db')
    study.optimize(f, n_trials=3000)
    print('params:', study.best_params)

if __name__ == '__main__':
    main()
