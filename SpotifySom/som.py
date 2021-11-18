import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import plot, show, pcolor, colorbar, bone

directory = '/home/dsdravos/Documents/MachineLearning_Spotify/Data/'

data_blues = pd.read_csv(directory+'blues_music_data.csv').iloc[:,5:16]
data_metal = pd.read_csv(directory+'metal_music_data.csv').iloc[:,5:16]

data_blues['target'] = '0'
data_metal['target'] = '1'

data_music = pd.concat([data_blues, data_metal], ignore_index=True)

features = data_music.loc[:,'danceability':'tempo']
targets = data_music['target']
dataTransform = MinMaxScaler().fit_transform(features)

for epochs in [100, 200, 300, 1000]:
    som = MiniSom(15, 15, 11, sigma=1.0, learning_rate=0.001)
    som.random_weights_init(dataTransform)

    som.train_random(dataTransform, epochs)

    bone()
    pcolor(som.distance_map().T)
    colorbar()  
    
    t = targets
    
    markers = ['o', 's']
    colors = ['r', 'b']
    
    for cnt, value in enumerate(dataTransform):
        w = som.winner(value)
      
        plot(
            w[0]+.5, 
            w[1]+.5, 
            markers[int(t[cnt])], 
            markerfacecolor='None', 
            markeredgecolor=colors[int(t[cnt])], 
            markersize=10, 
            markeredgewidth=2
        )

    show()