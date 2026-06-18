# 🍿 Movie Recommender System

A content-based movie recommender system that suggests movies similar to your favorites. 
This project uses the TMDB 5000 Movies Dataset, Natural Language Processing (NLP) for text vectorization, and Cosine Similarity to find the closest matches. 

The application is built with Python and deployed using an interactive Streamlit web interface, dynamically fetching high-resolution movie posters directly from the TMDB API.

## Features
- **Content-Based Filtering**: Recommends movies based on genres, keywords, cast, and crew.
- **Dynamic Poster Fetching**: Uses the TMDB API to display the official poster for every recommended movie.
- **On-the-Fly Computation**: Computes similarity matrices efficiently at runtime to bypass massive file storage requirements.
- **Sleek UI**: Custom dark-themed CSS and interactive Streamlit components.

## Tech Stack
- **Data Processing**: Pandas, NumPy
- **NLP & Machine Learning**: Scikit-Learn (CountVectorizer, Cosine Similarity), NLTK (PorterStemmer)
- **Web App**: Streamlit
- **API**: The Movie Database (TMDB) API

## Running Locally

To run this application on your local machine, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   cd YOUR_REPOSITORY
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your TMDB API Key**
   - Open `app.py`.
   - Locate the `fetch_poster()` function.
   - Replace the `api_key` parameter in the URL with your own TMDB API key.

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
   *The app will automatically launch in your browser at `http://localhost:8501`.*

## Dataset
The original dataset used for training and processing is the [TMDB 5000 Movie Dataset from Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
