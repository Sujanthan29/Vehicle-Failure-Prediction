# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 07:08:57 2024

@author: sujan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from feature_engine.outliers import Winsorizer
import statsmodels.api as sm
import scipy.stats as stats
import sweetviz
from scipy.stats import skew
from scipy.stats import kurtosis
import pickle
# MySQL database connection
from sqlalchemy import create_engine
from urllib.parse import quote

# Database connection details
user = 'root'  # user name
pw = 'user1'  # password
db = 'breakdown'  # database

# Creating engine to connect to the database
engine = create_engine(f"mysql+pymysql://{user}:{quote(pw)}@localhost/{db}")

# Read the data from the database
sql = 'select * from prediction'
prediction = pd.read_sql_query(sql, con=engine)

# Renaming columns
prediction.rename(columns={'Engine rpm': 'Engine_rpm', 'Lub oil pressure': 'Lub_oil_pressure',
                           'Fuel pressure': 'Fuel_pressure', 'Coolant pressure': 'Coolant_pressure',
                           'lub oil temp': 'lub_oil_temp', 'Coolant temp': 'Coolant_temp',
                           'Engine Condition': 'Engine_Condition'}, inplace=True)

# EDA
prediction.info()
prediction.describe()

## Measures of Central Tendency
prediction.mean()
prediction.median()
prediction.mode()

## Measures of Dispersion
prediction.std()
prediction.var()

## Measure of Assymetry
skew = skew(prediction)
skew

## Measure of peakness
kurtosis = kurtosis(prediction)
kurtosis


# Data Visualization
sns.pairplot(prediction)
plt.show()

##Bar plot
plt.bar(height=prediction.Engine_rpm, x= np.arange(1, 19536, 1))
plt.bar(height=prediction.Lub_oil_pressure, x= np.arange(1, 19536, 1))
plt.bar(height=prediction.Fuel_pressure, x= np.arange(1, 19536, 1))
plt.bar(height=prediction.Coolant_pressure, x= np.arange(1, 19536, 1))
plt.bar(height=prediction.lub_oil_temp, x= np.arange(1, 19536, 1))
plt.bar(height=prediction.Coolant_temp, x= np.arange(1, 19536, 1))
plt.bar(height=prediction.Engine_Condition, x= np.arange(1, 19536, 1))

## Histogram Plot
plt.hist(prediction.Engine_rpm, color='orange', edgecolor = 'black')
plt.hist(prediction.Lub_oil_pressure, color='orange', edgecolor = 'black')
plt.hist(prediction.Fuel_pressure, color='orange', edgecolor = 'black')
plt.hist(prediction.Coolant_pressure, color='orange', edgecolor = 'black')
plt.hist(prediction.lub_oil_temp, color='orange', edgecolor = 'black')
plt.hist(prediction.Coolant_temp, color='orange', edgecolor = 'black')
plt.hist(prediction.Engine_Condition, color='orange', edgecolor = 'black')

# Correlation Matrix
corr = prediction.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, cmap='YlGnBu', annot=True)
plt.title("Correlation Matrix")
plt.show()

# Normal Quantile-Quantile plot
sm.qqplot(prediction['Engine_rpm'], line ='s')
plt.show()

# Data Transformation
transformed_data, lmbda = stats.boxcox(prediction['Engine_rpm'])


# Sweetviz Auto EDA
report = sweetviz.analyze(prediction)
report.show_html('Report.html')

# Data Preprocessing
X = prediction.drop(columns=['Engine_Condition'])
y = prediction['Engine_Condition']



# Handling Missing Values
numeric_features = X.select_dtypes(exclude=['object']).columns
num_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='mean'))])
preprocessor = ColumnTransformer(transformers=[('mean', num_pipeline, numeric_features)])
impute_data = preprocessor.fit(X)
## Save the data preprocessing pipeline
joblib.dump(impute_data, 'impute.pkl')
X1 = pd.DataFrame(impute_data.transform(X), columns= X.columns)


# Outlier Treatment

## Box plot
X1.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
plt.subplots_adjust(wspace = 0.75) 
plt.show()
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=list(X1.columns))
outlier_pipeline = Pipeline(steps=[('winsor', winsor)])
preprocessor1 = ColumnTransformer(transformers=[('wins', outlier_pipeline, numeric_features)],
                                  remainder='passthrough')
winz_data = preprocessor1.fit(X)
## Save the data preprocessing pipeline
joblib.dump(winz_data, 'winzor.pkl')
X2 = pd.DataFrame(winz_data.transform(X), columns= X.columns)
X2.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
plt.subplots_adjust(wspace = 0.75) 
plt.show()

# Scaling with MinMaxScaler
scale_pipeline = Pipeline(steps=[('scale', MinMaxScaler())])
preprocessor2 = ColumnTransformer(transformers=[('scale', scale_pipeline, numeric_features)])
scale = preprocessor2.fit(X)
## Saving the data processing pipeline
joblib.dump(scale, 'scale.pkl')
X3 = pd.DataFrame(scale.transform(X), columns= X.columns)

# Splitting data into Train and Test sets
train_X, test_X, train_y, test_y = train_test_split(X3, y, test_size=0.2, stratify=y)


# Model Building
from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(train_X, test_X, train_y, test_y)

print(models)

from sklearn.model_selection import GridSearchCV

### KNeigbour Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
# 1. Instantiate the KNeighbours Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed
# 2. Fit the model to the training data
knn_classifier.fit(train_X, train_y)
# 3. Predict using the fitted model
train_predictions = knn_classifier.predict(train_X)
test_predictions = knn_classifier.predict(test_X)
# 4. Evaluate the model
train_accuracy = accuracy_score(train_y, train_predictions)
test_accuracy = accuracy_score(test_y, test_predictions)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
# Classification report
print("\nTraining Classification Report:")
print(classification_report(train_y, train_predictions))

print("\nTesting Classification Report:")
print(classification_report(test_y, test_predictions))

### 1. LGBM CLassifier
#pip install lightgbm
from lightgbm import LGBMClassifier
lgbm_clf = LGBMClassifier(random_state=42)
lgbm_clf.fit(train_X, train_y)
score_lgbm = lgbm_clf.score(test_X, test_y)
print(f"LGBM Classifier Score: {score_lgbm}")
# Define the hyperparameters to tune
params = {
    'learning_rate': [0.01, 0.1, 0.5],
    'num_leaves': [10, 20, 30],
    'max_depth': [5, 10, 15],
    'n_estimators': [50, 100, 200]}
# Initialize the LGBMClassifier
lgbm_clf = LGBMClassifier(random_state=42)
# Initialize GridSearchCV
grid_search1 = GridSearchCV(estimator=lgbm_clf, param_grid=params, cv=5, scoring='f1', verbose=2, n_jobs=-1)
# Fit GridSearchCV to the training data
grid_search1.fit(train_X, train_y)
# Print the best hyperparameters and the corresponding test score
print(f"Best Hyperparameters: {grid_search1.best_params_}")
print(f"Test Score: {grid_search1.best_score_:.4f}")


### 2. AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(random_state=42)
ada_clf.fit(train_X, train_y)
score_ada = ada_clf.score(test_X, test_y)
print(f"AdaBoost Classifier Score: {score_ada}")
# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.1, 0.5, 1],
}
# Initialize the AdaBoost Classifier
ada_boost = AdaBoostClassifier(random_state=42)
# Initialize GridSearchCV
grid_search2 = GridSearchCV(estimator=ada_boost, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
# Fit GridSearchCV to the training data
AB_new = grid_search2.fit(train_X, train_y)
# Extract the best parameters
best_params = grid_search2.best_params_
print(f"Best hyperparameters: {best_params}")
print(f"Test Score: {grid_search2.best_score_:.4f}")
from sklearn.metrics import confusion_matrix
# Predictions on test data using AdaBoost classifier without hyperparameter tuning
pred_ada = ada_clf.predict(test_X)
pred_ada
# Construct confusion matrix
conf_matrix_ada = confusion_matrix(test_y, pred_ada)
print("Confusion Matrix for AdaBoost Classifier:")
print(conf_matrix_ada)


### 3. ExtraTree Classifier
from sklearn.ensemble import ExtraTreesClassifier
et_clf = ExtraTreesClassifier(random_state=42)
et_clf.fit(train_X, train_y)
score_et = et_clf.score(test_X, test_y)
print(f"ExtraTrees Classifier Score: {score_et}")
# Create a dictionary of hyperparameters to tune
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]}
# Create an ExtraTreesClassifier with default hyperparameters
clf = ExtraTreesClassifier(random_state=0)
# Perform grid search with cross-validation
grid_search3 = GridSearchCV(clf, param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
grid_search3.fit(train_X, train_y)
# Print the best hyperparameters and the best score
print("Best hyperparameters: ", grid_search3.best_params_)
print("Best score: ", grid_search3.best_score_)


### 4. Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(random_state=42)
lr_clf.fit(train_X, train_y)
score_lr = lr_clf.score(test_X, test_y)
print(f"Logistic Regression Score: {score_lr}")
# Create a dictionary of hyperparameters to tune
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [100, 500, 1000]}
# Create a LogisticRegression object with default hyperparameters
lr_clf = LogisticRegression(random_state=42)
# Perform grid search with cross-validation
grid_search4 = GridSearchCV(lr_clf, param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
grid_search4.fit(train_X, train_y)
# Print the best hyperparameters and the best score
print("Best hyperparameters: ", grid_search4.best_params_)
print("Best score: ", grid_search4.best_score_)
# Fit the LogisticRegression model with the best hyperparameters
best_lr_clf = LogisticRegression(**grid_search4.best_params_, random_state=42)
best_lr_clf.fit(train_X, train_y)
# Evaluate the LogisticRegression model with the best hyperparameters
score_lr = best_lr_clf.score(test_X, test_y)
print(f"Logistic Regression Score: {score_lr}")


### 5. LinearDiscriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(train_X, train_y)
score_lda = lda_clf.score(test_X, test_y)
print(f"Linear Discriminant Analysis Score: {score_lda}")
# Create a dictionary of hyperparameters to tune
param_grid = {
    'shrinkage': ['auto', None],
    'tol': [1e-4, 1e-5, 1e-6],
    'n_components': [None, 2, 3]}
# Create a LinearDiscriminantAnalysis object with default hyperparameters
lda_clf = LinearDiscriminantAnalysis()
# Perform grid search with cross-validation
grid_search5 = GridSearchCV(lda_clf, param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
grid_search5.fit(train_X, train_y)
# Print the best hyperparameters and the best score
print("Best hyperparameters: ", grid_search5.best_params_)
print("Best score: ", grid_search5.best_score_)
# Fit the LinearDiscriminantAnalysis model with the best hyperparameters
best_lda_clf = LinearDiscriminantAnalysis(**grid_search5.best_params_)
best_lda_clf.fit(train_X, train_y)
# Evaluate the LinearDiscriminantAnalysis model with the best hyperparameters
score_lda = best_lda_clf.score(test_X, test_y)
print(f"Linear Discriminant Analysis Score: {score_lda}")


### 6. Calibratted Classifier CV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# Initialize an SVM classifier
svm_clf = SVC(random_state=42)
# Calibrate the SVM classifier using CalibratedClassifierCV
cccv_clf = CalibratedClassifierCV(svm_clf)
# Fit the calibrated classifier on the training data
cccv_clf.fit(train_X, train_y)
# Calculate the accuracy score of the calibrated classifier on the test data
score_cccv = cccv_clf.score(test_X, test_y)
# Print the accuracy score of the calibrated classifier
print(f"Calibrated Classifier CV Score: {score_cccv}")
# Define a dictionary of hyperparameters to tune for the base SVM estimator
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']}
# Perform grid search with cross-validation for the base SVM estimator
grid_search_svm = GridSearchCV(estimator=svm_clf, param_grid=param_grid_svm, cv=5, scoring='f1', verbose=2, n_jobs=-1)
grid_search_svm.fit(train_X, train_y)
# Print the best hyperparameters and the best score for the base SVM estimator
print("Best hyperparameters for base SVM estimator: ", grid_search_svm.best_params_)
print("Best score for base SVM estimator: ", grid_search_svm.best_score_)
# Get the best estimator from the grid search
best_svm_clf = grid_search_svm.best_estimator_
# Calibrate the best SVM classifier using CalibratedClassifierCV
best_cccv_clf = CalibratedClassifierCV(best_svm_clf)
# Fit the calibrated classifier on the training data
best_cccv_clf.fit(train_X, train_y)
# Evaluate the calibrated classifier on the test data
score_cccv_best = best_cccv_clf.score(test_X, test_y)
# Print the accuracy score of the calibrated classifier with the best hyperparameters
print(f"Calibrated Classifier CV Score with Best Hyperparameters: {score_cccv_best}")



### 7. LinearSVC
from sklearn.svm import LinearSVC
lsvc_clf = LinearSVC(random_state=42)
lsvc_clf.fit(train_X, train_y)
score_lsvc = lsvc_clf.score(test_X, test_y)
print(f"LinearSVC Score: {score_lsvc}")
# Create a dictionary of hyperparameters to tune
param_grid = {
    'C': [0.1, 1, 10, 100],
    'loss': ['squared_hinge', 'hinge'],
    'dual': [True, False],
    'tol': [1e-4, 1e-5, 1e-6],
    'multi_class': ['ovr', 'crammer_singer']}
# Create a LinearSVC object with default hyperparameters
lsvc_clf = LinearSVC(random_state=42)
# Perform grid search with cross-validation
grid_search7 = GridSearchCV(lsvc_clf, param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
grid_search7.fit(train_X, train_y)
# Print the best hyperparameters and the best score
print("Best hyperparameters: ", grid_search7.best_params_)
print("Best score: ", grid_search7.best_score_)
# Fit the LinearSVC model with the best hyperparameters
best_lsvc_clf = LinearSVC(**grid_search7.best_params_, random_state=42)
best_lsvc_clf.fit(train_X, train_y)
# Evaluate the LinearSVC model with the best hyperparameters
score_lsvc = best_lsvc_clf.score(test_X, test_y)
print(f"LinearSVC Score: {score_lsvc}")


### 8. Ridge Classifier
from sklearn.linear_model import RidgeClassifier
rc_clf = RidgeClassifier(random_state=42)
rc_clf.fit(train_X, train_y)
score_rc = rc_clf.score(test_X, test_y)
print(f"RidgeClassifier Score: {score_rc}")
# Create a dictionary of hyperparameters to tune
param_grid = {
    'alpha': [0.1, 1, 10, 100],
    'class_weight': [None, 'balanced'],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
# Create a RidgeClassifier object with default hyperparameters
rc_clf = RidgeClassifier(random_state=42)
# Perform grid search with cross-validation
grid_search8 = GridSearchCV(rc_clf, param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
grid_search8.fit(train_X, train_y)
# Print the best hyperparameters and the best score
print("Best hyperparameters: ", grid_search8.best_params_)
print("Best score: ", grid_search8.best_score_)
# Fit the RidgeClassifier model with the best hyperparameters
best_rc_clf = RidgeClassifier(**grid_search8.best_params_, random_state=42)
best_rc_clf.fit(train_X, train_y)
# Evaluate the RidgeClassifier model with the best hyperparameters
score_rc = best_rc_clf.score(test_X, test_y)
print(f"RidgeClassifier Score: {score_rc}")


### 9. SVC
from sklearn.svm import SVC
svm_clf = SVC(random_state=42)
svm_clf.fit(train_X, train_y)
score_svm = svm_clf.score(test_X, test_y)
print(f"SVC Score: {score_svm}")
# Create a dictionary of hyperparameters to tune
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly'],
    'degree': [2, 3, 4],
    'coef0': [0, 1],
    'shrinking': [True, False],
    'probability': [True, False],
    'tol': [1e-4, 1e-5, 1e-6],
    'cache_size': [200, 500],
    'class_weight': [None, 'balanced'],
    'max_iter': [-1, 100, 500],
    'decision_function_shape': ['ovr', 'ovo']}
# Create an SVC object with default hyperparameters
svm_clf = SVC(random_state=42)
# Perform grid search with cross-validation
grid_search9 = GridSearchCV(svm_clf, param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
grid_search9.fit(train_X, train_y)
# Print the best hyperparameters and the best score
print("Best hyperparameters: ", grid_search9.best_params_)
print("Best score: ", grid_search9.best_score_)
# Fit the SVC model with the best hyperparameters
best_svm_clf = SVC(**grid_search9.best_params_, random_state=42)
best_svm_clf.fit(train_X, train_y)
# Evaluate the SVC model with the best hyperparameters
score_svm =best_svm_clf.score(test_X, test_y)
print(f"SVC Score: {score_svm}")


### 10. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(train_X, train_y)
test_score = rf_clf.score(test_X, test_y)
print(f"Random Forest Classifier Test Score: {test_score:.4f}")

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search.fit(train_X, train_y)

# Print the best hyperparameters and the corresponding test score
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Test Score: {grid_search.best_score_:.4f}")


############### Auto ML########################
# 1. TPOT ( Tree based Pipeline Optimize Tool)
#pip install tpot
import tpot
from tpot import TPOTClassifier
# Initialize TPOT classifier
tpot = tpot.TPOTClassifier(generations=5, population_size=20, verbosity=2)
# Fit the model
tpot.fit(train_X, train_y)
# Make predictions
predictions = tpot.predict(test_X)

# 2. h2O automl
#pip install h2o
import h2o
# Initialize H2O
h2o.connect()
# Import data into H2O
train_h2o = h2o.H2OFrame(train_X)
test_h2o = h2o.H2OFrame(test_X)
train_h2o['Engine_Condition'] = h2o.H2OFrame(train_y)
# Create and train the H2O AutoML model
automl = H2OAutoML(max_models=5, seed=123)
automl.train(x=train_h2o.col_names, y='Engine_Condition', training_frame=train_h2o)
# Make predictions
predictions = automl.predict(test_h2o)['predict']

# 3. AutoSK Learn
pip install autosklearn
import autosklearn
import autosklearn.classification

# Initialize Auto-sklearn
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30)
# Fit Auto-sklearn to training data
automl.fit(train_X, train_y)
# Get the final ensemble estimator
ensemble_estimator = automl.automl_.get_models_with_weights()[-1][1]
# Print the ensemble estimator
print(ensemble_estimator)

# 4.Auto ML
!pip install auto_ml==2.9.9
from auto_ml import Predictor
# Initialize auto_ml Predictor
column_descriptions = {'Engine_Condition': 'output'}
ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
# Fit the Predictor
ml_predictor.train(train_X, model_names=['Random Forest', 'Decision Tree', 'Extra Trees'])
# Score the model
test_score = ml_predictor.score(test_X, test_y, verbose=0)
print("Test Score:", test_score)

# 5. PyCaret
pip install pycaret
from pycaret.classification import *
# Initialize PyCaret
exp1 = setup(data=pd.concat([train_X, train_y], axis=1), target='Engine_Condition', session_id=123)
# Compare models
best_model = compare_models(n_select=5)
# Tune models
tuned_best_model = [tune_model(model) for model in best_model]
# Blend models
blended_model = blend_models(estimator_list=tuned_best_model)

# Save the Best Model
Best_Score = AB_new.best_estimator_
pickle.dump(Best_Score, open('ab.pkl', 'wb'))

import os
os.getcwd()
