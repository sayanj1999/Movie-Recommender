import numpy as np
import pandas as pd
import ast
import pickle
import warnings
warnings.filterwarnings('ignore')
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('datasets/tmdb_5000_movies.csv')
credits = pd.read_csv('datasets/tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
movies = movies[['id','title', 'genres', 'keywords', 'overview','production_companies', 'cast', 'crew']]
movies.dropna(inplace=True)

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['production_companies'] = movies['production_companies'].apply(convert)

def extract_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 4:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(extract_cast)

def extract_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(extract_director)
movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['production_companies'] = movies['production_companies'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['production_companies'] + movies['cast'] + movies['crew']

movies_df = movies[['id','title','tags']]
movies_df['tags'] = movies_df['tags'].apply(lambda x:" ".join(x))
movies_df['tags'] = movies_df['tags'].apply(lambda x:x.lower())

ps = PorterStemmer()
def stem(word):
    y = []
    for i in word.split():
        y.append(ps.stem(i))
    return " ".join(y)

movies_df['tags'] = movies_df['tags'].apply(stem)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies_df['tags']).toarray()

similarity = cosine_similarity(vectors)

pickle.dump(movies_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
print("Models exported successfully to movie_dict.pkl and similarity.pkl")
