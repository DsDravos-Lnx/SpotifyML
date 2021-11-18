import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

directory = '/home/dsdravos/Documents/MachineLearning_Spotify/Data/'

data_blues = pd.read_csv(directory+'blues_music_data.csv').iloc[:,5:16]
data_metal = pd.read_csv(directory+'metal_music_data.csv').iloc[:,5:16]

data_blues['target'] = '0'
data_metal['target'] = '1'

data_music = pd.concat([data_blues, data_metal], ignore_index=True)

features = data_music.loc[:,'danceability':'tempo']
targets = data_music['target']

scaler = MinMaxScaler().fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train) 

y_pred = knn.predict(X_test)

confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label='1')
f1 = f1_score(y_test, y_pred, pos_label='1')
recall = recall_score(y_test, y_pred, pos_label='1')

print("Classe predita:   ", y_pred)
print("Classe verdadeira:", y_test)
print('Accuracy:', acc)
print('Precision:', prec)
print('f1:', f1)
print('recall:', recall)