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
