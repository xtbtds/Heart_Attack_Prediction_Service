import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Data preparation

print('Preparing data...')
df=pd.read_csv("heart.csv")

cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
num_cols = ["age","trtbps","chol","thalachh","oldpeak"]
target_col = ["output"]

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.output.values
y_test = df_test.output.values
del df_train['output']
del df_test['output']

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)
train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)


# Logistic Regression

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

parameters = {
    'penalty' : ['l1','l2'], 
    'C'       : np.logspace(-10,10,2),
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}

model_logreg = LogisticRegression()
print('Logistic regression training...')
scorings_list = ['accuracy', 'f1', 'roc_auc']
for scoring in scorings_list:
    clf = GridSearchCV(model_logreg,
                       param_grid = parameters,
                       scoring=scoring,
                       cv=10) 
    clf.fit(X_train,y_train)
    print("Tuned Hyperparameters :", clf.best_params_)
    print(f"{scoring} :",clf.best_score_)
    print("________________________")


# Random Forest

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

max_trees = 100

values = np.arange(max_trees) + 1

kf = KFold(n_splits=5, shuffle=True, random_state=1234)

global_scores = []

print("Random forest tuning with KFold...")
for train_indices, test_indices in tqdm(kf.split(X_train), total=5):
    scores = []
    
    X_train_kf = X_train[train_indices]
    y_train_kf = y_train[train_indices]

    X_test_kf = X_train[test_indices]
    y_test_kf = y_train[test_indices]
    
    forest = RandomForestClassifier(n_estimators=max_trees)
    forest.fit(X_train_kf, y_train_kf)
    trees = forest.estimators_
    
    for number_of_trees in tqdm(values, leave=False):
        tuned_forest = RandomForestClassifier(n_estimators=number_of_trees)
        
        tuned_forest.n_classes_ = 2
        tuned_forest.estimators_ = trees[:number_of_trees]

        scores.append(roc_auc_score(y_test_kf, tuned_forest.predict_proba(X_test_kf)[:, 1]))
    
    scores = np.array(scores)
    
    global_scores.append(scores)

global_scores = np.stack(global_scores, axis=0)
mean_cross_val_score = global_scores.mean(axis=0)

print("Random forest tuning with KFold - DONE")

# plt.figure(figsize=(15,8))
# plt.title('Random forest')

# plt.plot(values, 
#          mean_cross_val_score, 
#          label='mean values', 
#          color='red', 
#          lw=3)

# plt.xlabel('number of trees')
# plt.ylabel('roc-auc')
# plt.show()


# Tuned Random Forest

# from sklearn.model_selection import RandomizedSearchCV
print("Random forest tuning with GridSearch...")
grid = { 
    'n_estimators': np.linspace(100,1000,10, dtype=int),
    'max_features': ['sqrt', 'log2', 'auto'],
    'max_depth' : np.linspace(1,100,1),
    'criterion' :['gini', 'entropy']
}
rf_cv = GridSearchCV(estimator=RandomForestClassifier(), param_grid=grid, scoring='roc_auc',cv= 5)
rf_cv.fit(X_train, y_train)

rf_cv.best_params_, rf_cv.best_score_

# rf_best = RandomForestClassifier(criterion='gini',max_depth=65,max_features='sqrt',n_estimators=200)
rf_best = RandomForestClassifier(criterion='gini',max_depth=1,max_features='sqrt',n_estimators=300)
rf_best.fit(X_train, y_train)

dv = DictVectorizer(sparse=False)
test_dict = df_test.to_dict(orient='records')
X_test = dv.fit_transform(test_dict)

y_pred = rf_best.predict(X_test)
print("roc_auc_score",roc_auc_score(y_pred, y_test))
print("accuracy_score",accuracy_score(y_pred, y_test))


# Save model

print(f"saving model {rf_best}...")

import pickle

output_file = 'model.bin'

f_out = open(output_file,'wb')
pickle.dump((dv, rf_best),f_out)
f_out.close()

print("model saved")








