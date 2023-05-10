from google.colab import drive
drive.mount('/content/drive')

import os
from numpy.ma.extras import median
import numpy as np
from matplotlib.patches import Ellipse
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
import glob

#SPLIT PAtient
os.chdir('/content/drive/My Drive/laba/new_data/train')

path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))

train_data = pd.DataFrame()
for f in csv_files:
   print('File Name:', f.split("\\")[-1])
   # read the csv file
   df2 = pd.read_csv(f)
   df2.dropna(inplace=True)
   frames = [train_data , df2]
   train_data  = pd.concat(frames)


train_data  = train_data.reset_index(drop=True)   


#---------------------------------------
os.chdir('/content/drive/My Drive/laba/new_data/test')
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))

test_data = pd.DataFrame()
for f in csv_files:
   # read the csv file
   df2 = pd.read_csv(f)
   df2.dropna(inplace=True)
   frames = [test_data, df2]
   test_data = pd.concat(frames)

test_data = test_data.reset_index(drop=True)

def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 
            'kappa': kappa, 
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 
            'cm': cm}

#test 
#all_data['lable'] = all_data['lable'].apply(lambda x: 1 if x == 'yes' else 0)
feature = train_data.drop('class', axis=1)
#feature = feature.drop('lable_multy', axis=1)
#feature = feature.drop('TLX_mental', axis=1)
#feature = feature.drop('user', axis=1)
#feature = feature.drop('Unnamed: 0.1', axis=1)
feature = feature.drop(train_data.columns[0], axis=1)
X_train = feature
y_train = train_data['class']

feature_test = test_data.drop('class', axis=1)
#feature_test = feature_test.drop('lable_multy', axis=1)
#feature_test = feature_test.drop('user', axis=1)
#feature_test = feature_test.drop('Unnamed: 0.1', axis=1)
#feature_test = feature_test.drop('TLX_mental', axis=1)
feature_test = feature_test.drop(test_data.columns[0], axis=1)
X_test = feature_test
y_test = test_data ['class']

print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas import DataFrame

# defining parameters 
params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'binary',
    'num_leaves': 20,
    'learnnig_rage': 0.05,
    'metric': {'auc', 'binary_logloss'},
    'max_depth': {-5},
    'verbose': 5
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
model = lgb.train(params, train_set=lgb_train, valid_sets=lgb_eval, early_stopping_rounds=30)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score
print(y_pred)
y_pred=y_pred.round(0)
print(y_pred)
y_pred=y_pred.astype(int)
print(y_pred)
roc_auc_score(y_pred,y_test)

from sklearn import metrics
from sklearn.metrics import roc_curve, auc

# Predict Test Data 
acc = metrics.accuracy_score(y_test, y_pred)
acc = metrics.accuracy_score(y_test, y_pred)
prec = metrics.precision_score(y_test, y_pred)
rec = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
kappa = metrics.cohen_kappa_score(y_test, y_pred)
# Calculate area under curve (AUC)
fpr, tpr, th = roc_curve(y_test, y_pred)
auc_val = auc(fpr,tpr)
# Display confussion matrix
cm = metrics.confusion_matrix(y_test, y_pred)

#roc_auc_score metric
roc_auc = roc_auc_score(y_pred,y_test)
print('Accuracy {:.4f}'.format(acc))
print('Precision {:.4f}'.format(prec))
print('Recall {:.4f}'.format(rec))
print('F1 {:.4f}'.format(f1))

import xgboost as xgb
xg_reg = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
               alpha = 10, n_estimators = 10)
#xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
#                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
y_pred = xg_reg.predict(X_test)

import matplotlib.pyplot as plt
# Predict Test Data 
acc = metrics.accuracy_score(y_test, y_pred)
acc = metrics.accuracy_score(y_test, y_pred)
prec = metrics.precision_score(y_test, y_pred)
rec = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
kappa = metrics.cohen_kappa_score(y_test, y_pred)
# Calculate area under curve (AUC)
fpr, tpr, th = roc_curve(y_test, y_pred)
auc_val = auc(fpr,tpr)
# Display confussion matrix
cm = metrics.confusion_matrix(y_test, y_pred)

#roc_auc_score metric
roc_auc = roc_auc_score(y_pred,y_test)
print('Accuracy {:.4f}'.format(acc))
print('Precision {:.4f}'.format(prec))
print('Recall {:.4f}'.format(rec))
print('F1 {:.4f}'.format(f1))

from sklearn import tree
# Building Decision Tree model 
dtc = tree.DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
# Evaluate Model Decision Tree model 
dtc_eval = evaluate_model(dtc, X_test, y_test)


from sklearn.ensemble import RandomForestClassifier
# Building Random Forest model 
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
# Evaluate Model Random Forest model
rf_eval = evaluate_model(rf, X_test, y_test)


from sklearn.naive_bayes import GaussianNB
# Building Naive Bayes model 
nb = GaussianNB()
nb.fit(X_train, y_train)
# Evaluate Model Naive Bayes model 
nb_eval = evaluate_model(nb, X_test, y_test)

from sklearn.neighbors import KNeighborsClassifier
# Building KNN model 
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# Evaluate Model KNN model 
knn_eval = evaluate_model(knn, X_test, y_test)

# # Building model Logistic Regression
from sklearn.linear_model import LogisticRegression 
log = LogisticRegression() 
log.fit(X_train, y_train)
#y_pred = classifier.predict(x_test)
# Support svm Classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
svm = SVC()
clf = CalibratedClassifierCV(svm)
clf.fit(X_train, y_train)
# Evaluate Model Logistic Regression 
log_eval = evaluate_model(log, X_test, y_test)
# Support svm Classifier
clf_eval = evaluate_model(clf, X_test, y_test)


#Extreme Gradient Boosting
import xgboost as xgb
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train,y_train)
xgb_eval = evaluate_model(xgb_classifier, X_test, y_test)


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
# Train Adaboost Classifer
abc.fit(X_train, y_train)
abc_eval = evaluate_model(abc, X_test, y_test)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
fig.set_figheight(7)
fig.set_figwidth(14)
fig.set_facecolor('white')

barWidth = 0.1
dtc_score = [dtc_eval['acc'], dtc_eval['prec'], dtc_eval['rec'], dtc_eval['f1']]
rf_score = [rf_eval['acc'], rf_eval['prec'], rf_eval['rec'], rf_eval['f1']]
nb_score = [nb_eval['acc'], nb_eval['prec'], nb_eval['rec'], nb_eval['f1']]
knn_score = [knn_eval['acc'], knn_eval['prec'], knn_eval['rec'], knn_eval['f1']]
log_score = [log_eval['acc'], log_eval['prec'], log_eval['rec'], log_eval['f1']]
clf_score = [clf_eval['acc'], clf_eval['prec'], clf_eval['rec'], clf_eval['f1']]

xgb_score = [xgb_eval['acc'], xgb_eval['prec'], xgb_eval['rec'], xgb_eval['f1']]
abc_score = [abc_eval['acc'], abc_eval['prec'], abc_eval['rec'], abc_eval['f1']]

r1 = np.arange(len(dtc_score))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]
r8 = [x + barWidth for x in r7]


ax1.bar(r1, dtc_score, width=barWidth, edgecolor='white', label='Decision Tree')
ax1.bar(r2, rf_score, width=barWidth, edgecolor='white', label='Random Forest')
ax1.bar(r3, nb_score, width=barWidth, edgecolor='white', label='Naive Bayes')
ax1.bar(r4, knn_score, width=barWidth, edgecolor='white', label='K-Nearest Neighbors')
ax1.bar(r5, log_score, width=barWidth, edgecolor='white', label='Logistic Regression')
ax1.bar(r6, clf_score, width=barWidth, edgecolor='white', label='Vector Machine')
ax1.bar(r7, xgb_score, width=barWidth, edgecolor='white', label='XGBoost')
ax1.bar(r8, abc_score, width=barWidth, edgecolor='white', label='AdaBoost')

ax1.set_xlabel('Metrics', fontweight='bold')
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
ax1.set_xticks([r + (barWidth * 1.5) for r in range(len(dtc_score))], )
ax1.set_xticklabels(labels)
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_ylim(0, 1)
ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
ax1.legend()

ax2.plot(dtc_eval['fpr'], dtc_eval['tpr'], label='Decision Tree, auc = {:0.5f}'.format(dtc_eval['auc']))
ax2.plot(rf_eval['fpr'], rf_eval['tpr'], label='Random Forest, auc = {:0.5f}'.format(rf_eval['auc']))
ax2.plot(nb_eval['fpr'], nb_eval['tpr'], label='Naive Bayes, auc = {:0.5f}'.format(nb_eval['auc']))
ax2.plot(knn_eval['fpr'], knn_eval['tpr'], label='K-Nearest Nieghbor, auc = {:0.5f}'.format(knn_eval['auc']))
ax2.plot(log_eval['fpr'], log_eval['tpr'], label='Logistic Regression, auc = {:0.5f}'.format(log_eval['auc']))
ax2.plot(clf_eval['fpr'], clf_eval['tpr'], label='Vector Machine, auc = {:0.5f}'.format(clf_eval['auc']))
ax2.plot(xgb_eval['fpr'], xgb_eval['tpr'], label='XGBoost, auc = {:0.5f}'.format(xgb_eval['auc']))
ax2.plot(abc_eval['fpr'], abc_eval['tpr'], label='AdaBoost, auc = {:0.5f}'.format(abc_eval['auc']))


ax2.set_xlabel('False Positive Rate', fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontweight='bold')
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend(loc=4)

plt.show()

print('ACCURACY')
print ('Evaluate Model Decision Tree model ')
print (dtc_eval['acc']) 
print ('Evaluate Model Random Forest model')
print (rf_eval['acc']) 
print ('Evaluate Model Naive Bayes model ')
print (nb_eval['acc']) 
print ('Evaluate Model KNN model') 
print (knn_eval['acc']) 
print ('Evaluate Logistic Regression ') 
print (log_eval['acc']) 
print ('Evaluate svm Classifier') 
print (clf_eval['acc']) 
print ('Evaluate AdaBoostClassifier') 
print (abc_eval['acc']) 
print ('Evaluate Extreme Gradient Boosting') 
print (xgb_eval['acc']) 

print('prec')
print ('Evaluate Model Decision Tree model ')
print (dtc_eval['prec']) 
print ('Evaluate Model Random Forest model')
print (rf_eval['prec']) 
print ('Evaluate Model Naive Bayes model ')
print (nb_eval['prec']) 
print ('Evaluate Model KNN model') 
print (knn_eval['prec']) 
print ('Evaluate Logistic Regression ') 
print (log_eval['prec']) 
print ('Evaluate svm Classifier') 
print (clf_eval['prec']) 
print ('Evaluate AdaBoostClassifier') 
print (abc_eval['prec']) 
print ('Evaluate Extreme Gradient Boosting') 
print (xgb_eval['prec']) 

print('rec')
print ('Evaluate Model Decision Tree model ')
print (dtc_eval['rec']) 
print ('Evaluate Model Random Forest model')
print (rf_eval['rec']) 
print ('Evaluate Model Naive Bayes model ')
print (nb_eval['rec']) 
print ('Evaluate Model KNN model') 
print (knn_eval['rec']) 
print ('Evaluate Logistic Regression ') 
print (log_eval['rec']) 
print ('Evaluate svm Classifier') 
print (clf_eval['rec']) 
print ('Evaluate AdaBoostClassifier') 
print (abc_eval['rec']) 
print ('Evaluate Extreme Gradient Boosting') 
print (xgb_eval['rec']) 

print('f1')
print ('Evaluate Model Decision Tree model ')
print (dtc_eval['f1']) 
print ('Evaluate Model Random Forest model')
print (rf_eval['f1']) 
print ('Evaluate Model Naive Bayes model ')
print (nb_eval['f1']) 
print ('Evaluate Model KNN model') 
print (knn_eval['f1']) 
print ('Evaluate Logistic Regression ') 
print (log_eval['f1']) 
print ('Evaluate svm Classifier') 
print (clf_eval['f1']) 
print ('Evaluate AdaBoostClassifier') 
print (abc_eval['f1']) 
print ('Evaluate Extreme Gradient Boosting')
print (xgb_eval['f1']) 

print('auc')

print ('Evaluate Model Decision Tree model ')
print (dtc_eval['auc']) 
print ('Evaluate Model Random Forest model')
print (rf_eval['auc']) 
print ('Evaluate Model Naive Bayes model ')
print (nb_eval['auc']) 
print ('Evaluate Model KNN model') 
print (knn_eval['auc']) 
print ('Evaluate Logistic Regression ') 
print (log_eval['auc']) 
print ('Evaluate svm Classifier') 
print (clf_eval['auc']) 
print ('Evaluate AdaBoostClassifier') 
print (abc_eval['auc']) 
print ('Evaluate Extreme Gradient Boosting') 
print (xgb_eval['auc'])