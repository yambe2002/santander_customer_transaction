## Kaggle Sandanter Customer Transaction Prediction contest

### v.1
Simple logistic regression
 - SGDClassifier(loss='log', max_iter=50000, tol=1e-3, alpha=0.1)
 - local cv: 0.83363
 - lb: 0.838

### v.2
Try adding PCA to v.1 (simple logistic regression)
 - local cv: cv: 0.8168678702098181
##### NOT GOOD IDEA TO APPLY PCA

### v.3
Make some feature selections (simple logistic regression)
 - SGDClassifier(loss='log', max_iter=5000, tol=1e-7, alpha=0.01)
 - local cv: 0.8515980352095942

