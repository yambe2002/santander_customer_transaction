## Kaggle Sandanter Customer Transaction Prediction contest 
### v.1
Logistic regression
 - SGDClassifier(loss='log', max_iter=50000, tol=1e-3, alpha=0.1)
 - local cv: 0.83363
 - lb: 0.839

### v.2
Try adding PCA (logistic regression)
 - local cv: cv: 0.8168678702098181
##### NOT GOOD IDEA TO APPLY PCA

### v.3
Modify parameters (logistic regression)
 - SGDClassifier(loss='log', max_iter=5000, tol=1e-7, alpha=0.01)
 - local cv: 0.8515980352095942

### v.4
Apply std scaling (logistic regression)
 - SGDClassifier(loss='log', max_iter=5000, tol=1e-7, alpha=0.01)
 - local cv: 0.8595493966729728
 - lb: 0.861

### v.5
Modify parameters (logistic regression)
 - SGDClassifier(loss='log', max_iter=5000, tol=1e-7, alpha=0.3)
 - local cv: 0.8597059678850029

### v.6
Add Multi-layer perceptron
 - MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(5, 2), random_state=1)
 - local cv: 0.8543658603036868

### v.7
Add logic to fix data skew (not used yet)

### v.8
Add RandamForest
 - RandomForestClassifier(n_estimators=1000, criterion='entropy')
 - local cv: 8.83 (with data skew fix)

### v.9
Add voting

### v.10
Add RandamForest score (v.8)

### v.11
Ensenble
 - SGDClassifier(loss='log', max_iter=5000, tol=1e-7, alpha=0.3)
 - MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(5, 2), random_state=1)
 - RandomForestClassifier(n_estimators=100, criterion='entropy')
 - local cv: 0.87 (with data skew fix)
 - lb: 0.869

### v.12
 - Add GNB
 - local cv: 0.8874777137937799 (with data skew fix)

### v.13
 - Add LinearDiscriminantAnalysis() - 0.8608604117713252 (with data skew fix)
 - Add QuadraticDiscriminantAnalysis(tol=1e-12) - 0.820027075184469 (with data skew fix)
 - Add AdaBoostClassifier(lg) - 0.8616254422551686 (with data skew fix)
 - Add BaggingClassifier(lg) - 0.8614224931614438 (with data skew fix)

### v.14
 - Submit with GaussianNB
 - local cv: 0.8883828469044662
 - lb: 0.888

### v.15
 - Add LightGBM+kfold implementation

### v.16
 - Submit with LightGBM+kfold
 - lb: 0.899

### v.17
 - Add code of GridSearchCV

### v.18
 - Trial submit: LightBGM, max_depth=127
 - local cv: 0.90775 (by grid search)
 - lb: 0.891 (fold_n=2)

### v.19
 - LightGBM submission
 - lb: 0.899(n_split=10)

### v.20
 - n_split=5 with v.19
 - lb: 0.899

### v.21
 - try overfit case ('max_depth': 127)
 - lb: 0.896(n_split=10)

### v.21
 - add code to apply data skew fix

### v.22
 - another try with some param tuning
 - lb: 0.898

### v.23
 - v.20 with data skew fix
 - lb: 0.895

### v.24
 - another tryout with param tuning
 - local: 0.898
 - lb: 0.899

### v.25
 - param tuning
 - local: 0.9000721582954704
 - lb: 0.899 (best so far)

### v.26
 - same as v.25, but lower learning_rate (0.04 -> 0.01)
 - local: 0.9002425993336695
 - lb: 0.900 (best so far)

### v.27
 - param tuning
 - local: 0.9001731611321231
 - lb: 0.900

### v.28
 - 1st stacking trial
 - local: 1.0??
 - lb: 0.669 - something is wrong

### v.29
 - 2nd stacking trial
 - local: 1.0??
 - lb: 0.539 - something is wrong

### v.30
 - lgbm with params from other models
 - local: 0.8945364695860429
 - lb: 0.895

### v.31
 - lgbm with additional params without other models
 - local: 0.899363708306008
 - lb: not yet

### v.32
 - lgbm without k-fold (test_size:0.2), with additional params
 - local: 0.9013526252596797
 - lb: not yet

### v.33
 - lgbm without k-fold (test_size:0.2), without additional parms
 - local: 0.901284141805979
 - lb: not yet