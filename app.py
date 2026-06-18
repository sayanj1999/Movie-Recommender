import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config for premium look
st.set_page_config(page_title="Movie Recommender", page_icon="🍿", layout="wide")

# Custom CSS for modern design aesthetics
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Headers */
    h1 {
        color: #38bdf8;
        font-family: 'Inter', sans-serif;
        text-align: center;
        padding-bottom: 2rem;
    }
    
    /* Selectbox styling */
    .stSelectbox label {
        color: #cbd5e1 !important;
        font-size: 1.1rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #38bdf8;
        color: #0f172a;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        display: block;
        margin: 0 auto;
    }
    .stButton>button:hover {
        background-color: #7dd3fc;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    
    /* Movie title text */
    .movie-title {
        text-align: center;
        font-weight: bold;
        color: #f1f5f9;
        margin-top: 0.8rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=cb3bacb4aa4207f0b6f097d9af81dd58&language=en-US"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return "https://dummyimage.com/500x750/38bdf8/0f172a.png&text=API+Key+Error"
            
        data = response.json()
        if 'poster_path' in data and data['poster_path'] is not None:
            return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
        return "https://dummyimage.com/500x750/cbd5e1/0f172a.png&text=No+Poster"
    except:
        return "https://dummyimage.com/500x750/ef4444/0f172a.png&text=Error"

@st.cache_data
def load_data():
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    
    # Compute similarity on the fly to avoid large file uploads
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    
    return movies, similarity

st.title('🍿 Movie Recommender System')

try:
    movies, similarity = load_data()
except Exception as e:
    st.error(f"Could not load the recommendation models. Error: {e}")
    st.stop()

def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        # Get top 5 most similar movies
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        recommended_movies_posters = []
        
        for i in movies_list:
            movie_id = movies.iloc[i[0]].id
            recommended_movies.append(movies.iloc[i[0]].title)
            recommended_movies_posters.append(fetch_poster(movie_id))
            
        return recommended_movies, recommended_movies_posters
    except Exception as e:
        print(e)
        return [], []

# Layout
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    selected_movie_name = st.selectbox(
        'Select or type a movie you like:',
        movies['title'].values
    )
    
    st.write("") # Spacer
    recommend_clicked = st.button('Get Recommendations')

# Show recommendations
if recommend_clicked:
    with st.spinner('Finding the perfect movies for you...'):
        names, posters = recommend(selected_movie_name)
        
        if names:
            st.write("---")
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    # Display poster
                    st.image(posters[i], use_container_width=True)
                    # Display title
                    st.markdown(f"<div class='movie-title'>{names[i]}</div>", unsafe_allow_html=True)
        else:
            st.error("Oops! Something went wrong finding recommendations for this movie.")
