import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


directory = '/home/dsdravos/Documents/MachineLearning_Spotify/Data/'

data_blues = pd.read_csv(directory+'blues_music_data.csv').iloc[:,5:16]
data_metal = pd.read_csv(directory+'metal_music_data.csv').iloc[:,5:16]

data_blues['target'] = '0'
data_metal['target'] = '1'

data_music = pd.concat([data_blues, data_metal], ignore_index=True)

features = data_music.loc[:,'danceability':'tempo']
targets = data_music['target']

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)

parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 5]}

svm = SVC()

clf = GridSearchCV(svm, parameters)
clf.fit(X_train, y_train)

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

y_pred = clf.predict(X_test)

for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("mean: %0.3f std: (+/-%0.03f) for %r" % (mean, std * 2, params))

print('\n')    
print('Best params founder:', clf.best_params_)     
print('Score for best params:', clf.best_score_)
print('\n')    
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label='1')
f1 = f1_score(y_test, y_pred, pos_label='1')
recall = recall_score(y_test, y_pred, pos_label='1')

print("Accuracy:", acc)
print("Precision: ", prec)
print("F1:", f1)
print("Recall:", recall)