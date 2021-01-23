import pickle
import numpy as np, pandas as pd 
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from nturl2path import url2pathname, pathname2url
data_artist = pd.read_csv('data/data_by_artist.csv')
data_artist.drop(['duration_ms','key','mode','count'],axis=1,inplace=True)
data_artist['popularity'] = data_artist['popularity']/100
data_artist['tempo'] = (data_artist['tempo'] - 50)/100
data_artist['loudness'] = (data_artist['loudness'] + 60)/60
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 
            'loudness', 'speechiness', 'tempo', 'valence', 'popularity']

def choice_of_user(artistRatingDict):
    artists = artistRatingDict.keys()
    artist_Mat = data_artist[data_artist['artists'].isin(artists)]
    #print(artist_Mat)
    
    for artist, rating in artistRatingDict.items():
        artist_Mat.loc[artist_Mat['artists']== artist,features] = artist_Mat.loc[artist_Mat['artists'] ==artist,features]#.mul(rating,axis=0)
    
  
    userProfile = artist_Mat.loc[:,features].sum(axis=0)
    normalized_userProfile = (userProfile/userProfile.sum())*10
    
    return normalized_userProfile

def createRecomMatrix(userProfile,artists):
        
    artist_Mat = data_artist[~data_artist['artists'].isin(artists)]
    artist_Mat.set_index('artists',inplace=True) 
    #print(userProfile)
    #print(artist_Mat.head())
        
    recomMat = pd.DataFrame(artist_Mat.values*userProfile.values,columns=artist_Mat.columns, index=artist_Mat.index)
    recomMat = recomMat.sum(axis=1)
    recomMat.sort_values(ascending = False,inplace=True)
        
    return recomMat

def recommend(artistRatingDict):
        
    userProfile = choice_of_user(artistRatingDict)
        
    recommendationMat = createRecomMatrix(userProfile,artistRatingDict.keys()) 
        
    return recommendationMat.head(5)
app = Flask(__name__)
pickle.dump(recommend, open('model.pkl','wb'))
model = pickle.load(open('model/model.pkl','rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [x for x in request.form.values()]
    dictionary={data[0]:data[1]}

    
    prediction = model(dictionary).to_dict()
    return render_template('index.html', predict = prediction)

if __name__ == '__main__':
    app.run(debug=True)
