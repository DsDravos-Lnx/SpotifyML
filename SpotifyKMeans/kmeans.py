import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import preprocessing
from sklearn.cluster import KMeans

directory = '/home/dsdravos/Documents/MachineLearning_Spotify/Data/'

df = pd.read_csv(directory + 'final_audio_features.csv')
df1 = pd.read_csv(directory + 'final_audio_features.csv')
df1.drop(['duration_ms', 'uri', 'genre', 'type', 'mode', 'key'], axis=1, inplace=True)

df1 = df[['instrumentalness', 'speechiness', 'valence']]

x = df1.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled2 = min_max_scaler.fit_transform(x)
df1 = pd.DataFrame(x_scaled2)

kmeans = KMeans(init='k-means++',
                n_clusters=4,
                random_state=15,
                max_iter = 500).fit(x_scaled2)

df1['kmeans'] = kmeans.labels_
df1.columns = ['energy', 'instrumentalness', 'loudness', 'kmeans']

kmeans = df1['kmeans']
df['kmeans'] = kmeans

fig = px.scatter_3d(df, x='energy', y='instrumentalness', z='loudness', color='kmeans')
fig.show()


c0 = df1[df1['kmeans']==0]
c1 = df1[df1['kmeans']==1]
c2 = df1[df1['kmeans']==2]
c3 = df1[df1['kmeans']==3]

c0.drop(['kmeans'], axis=1, inplace=True)
c1.drop(['kmeans'], axis=1, inplace=True)
c2.drop(['kmeans'], axis=1, inplace=True)
c3.drop(['kmeans'], axis=1, inplace=True)


x = c0.values 
min_max_scaler = preprocessing.MinMaxScaler()
c0_scaled = min_max_scaler.fit_transform(x)
c0 = pd.DataFrame(c0_scaled)
c0.columns = ['energy', 'instrumentalness', 'loudness' ]
c0=c0.melt(var_name='groups', value_name='vals')

x = c1.values
min_max_scaler = preprocessing.MinMaxScaler()
c1_scaled = min_max_scaler.fit_transform(x)
c1 = pd.DataFrame(c1_scaled)
c1.columns = ['energy', 'instrumentalness', 'loudness' ]
c1=c1.melt(var_name='groups', value_name='vals')

x = c2.values 
min_max_scaler = preprocessing.MinMaxScaler()
c2_scaled = min_max_scaler.fit_transform(x)
c2 = pd.DataFrame(c2_scaled)
c2.columns = ['energy', 'instrumentalness', 'loudness']
c2=c2.melt(var_name='groups', value_name='vals')

x = c3.values 
min_max_scaler = preprocessing.MinMaxScaler()
c3_scaled = min_max_scaler.fit_transform(x)
c3 = pd.DataFrame(c3_scaled)
c3.columns = ['energy', 'instrumentalness', 'loudness']
c3=c3.melt(var_name='groups', value_name='vals')

f, axes = plt.subplots(4, 1)
ax = sns.violinplot( data=c0 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[0])
ax = sns.violinplot( data=c1 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[1])
ax = sns.violinplot( data=c2 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[2])
ax = sns.violinplot( data=c3 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[3])

plt.show()
