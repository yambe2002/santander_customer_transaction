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


n_splits = 5
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def objective(X, y, trial):
    score = 0.0

    params = {
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
    }

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)
            
        model = lgb.train(params,train_data,num_boost_round=20000,
                        valid_sets = [train_data, valid_data],verbose_eval=300,early_stopping_rounds = 100)
        
        score += model.best_score /n_splits
    
    return 1.0 - score


def main():
    f = partial(objective, X, y)
    study = optuna.create_study()
    study.optimize(f, n_trials=100)
    print('params:', study.best_params)

if __name__ == '__main__':
    main()
