import streamlit as st
import pandas as pd
import joblib
import base64
import numpy as np

# Load the saved IMDB Score model
model_path = 'C:\\Users\\Sugam Sharma\\Desktop\\Project Box Office\\ML_MODEL\\imdb_score_model.pkl'
model = joblib.load(model_path)

# Define the expected columns for the IMDB Score model
expected_columns = [
    'num_critic_for_reviews', 'duration', 'director_facebook_likes',
    'actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross',
    'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster',
    'num_user_for_reviews', 'budget', 'title_year',
    'actor_2_facebook_likes', 'aspect_ratio', 'movie_facebook_likes'
]

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    st.set_page_config(layout="wide")

    # Set the title of the web app with a color of your choice
    st.markdown(
        "<h1 style='text-align: center; color: #ff6666;'>🎬 Box Office Revenue Prediction 🎬</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h2 style='text-align: center; color: #ff6666; padding: 10px; border-radius: 10px;'>Enter the movie characteristics to predict its IMDB Score and Box Office Revenue:</h2>",
        unsafe_allow_html=True
    )

    # Convert image to base64 and use as background
    bg_image_base64 = get_base64_image('assets/bgimg.png')

    # Apply background image and style
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{bg_image_base64});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: #333333;
            font-family: 'Arial', sans-serif;
        }}
        .stButton>button {{
            background-color: #800000;
            color: #FFFFFF;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #A52A2A;
        }}
        .stSlider>div {{
            color: #FFFFFF;
        }}
        .stSlider>div>div>div>input {{
            background: linear-gradient(90deg, #800000, #A52A2A);
            color: #FFFFFF;
            border-radius: 5px;
        }}
        .stTextInput>div>input {{
            background-color: #FFFFFF;
            color: #333333;
            border: 1px solid #800000;
            border-radius: 5px;
            padding: 10px;
        }}
        .stSelectbox>div>div>div>select {{
            background-color: #FFFFFF;
            color: #333333;
            border: 1px solid #800000;
            border-radius: 5px;
            padding: 10px;
        }}
        .stSubheader, .stMarkdown {{
            color: #333333;
            background-color: transparent;
            padding: 10px;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create a two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader('Movie Characteristics')

        # Add input fields for features with improved text and help messages
        num_critic_for_reviews = st.number_input('Number of Critic Reviews', 0, 1000, 100, help="Number of critic reviews for the movie.")
        duration = st.slider('Duration (minutes)', 30, 300, 120, help="Duration of the movie in minutes.")
        director_facebook_likes = st.number_input('Director Facebook Likes', 0, 1000000, 1000, help="Number of Facebook likes for the director.")
        actor_3_facebook_likes = st.number_input('Actor 3 Facebook Likes', 0, 1000000, 1000, help="Number of Facebook likes for Actor 3.")
        actor_1_facebook_likes = st.number_input('Actor 1 Facebook Likes', 0, 1000000, 1000, help="Number of Facebook likes for Actor 1.")
        gross = st.number_input('Gross Revenue', 0, 1000000000, 5000000, help="Gross revenue of the movie.")
        num_voted_users = st.number_input('Number of Voted Users', 0, 1000000, 1000, help="Number of users who voted for the movie.")
        cast_total_facebook_likes = st.number_input('Cast Total Facebook Likes', 0, 10000000, 500000, help="Total Facebook likes for all cast members.")
        facenumber_in_poster = st.number_input('Face Number in Poster', 0, 10, 1, help="Number of faces in the movie poster.")
        num_user_for_reviews = st.number_input('Number of User Reviews', 0, 1000000, 1000, help="Number of user reviews for the movie.")
        budget = st.number_input('Budget', 0, 1000000000, 10000000, help="Budget for the movie.")
        title_year = st.slider('Title Year', 1900, 2024, 2020, help="Year the movie was released.")
        actor_2_facebook_likes = st.number_input('Actor 2 Facebook Likes', 0, 1000000, 1000, help="Number of Facebook likes for Actor 2.")
        aspect_ratio = st.number_input('Aspect Ratio', 0.5, 2.0, 1.78, step=0.01, help="Aspect ratio of the movie.")
        movie_facebook_likes = st.number_input('Movie Facebook Likes', 0, 1000000, 5000, help="Number of Facebook likes for the movie.")

        # Prepare input data as a DataFrame
        input_data = pd.DataFrame({
            'num_critic_for_reviews': [num_critic_for_reviews],
            'duration': [duration],
            'director_facebook_likes': [director_facebook_likes],
            'actor_3_facebook_likes': [actor_3_facebook_likes],
            'actor_1_facebook_likes': [actor_1_facebook_likes],
            'gross': [gross],
            'num_voted_users': [num_voted_users],
            'cast_total_facebook_likes': [cast_total_facebook_likes],
            'facenumber_in_poster': [facenumber_in_poster],
            'num_user_for_reviews': [num_user_for_reviews],
            'budget': [budget],
            'title_year': [title_year],
            'actor_2_facebook_likes': [actor_2_facebook_likes],
            'aspect_ratio': [aspect_ratio],
            'movie_facebook_likes': [movie_facebook_likes]
        })

        # Ensure columns are in the same order as during model training
        input_data = input_data[expected_columns]

    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            with st.spinner('Predicting...'):
                # Predict IMDB Score
                imdb_score = model.predict(input_data)[0]

                # Assuming a simple linear relationship for Box Office Revenue
                # Replace with actual model if available
                # Example linear relationship: revenue = intercept + coefficient * imdb_score
                intercept = 5000000  # Example intercept
                coefficient = 10000000  # Example coefficient
                box_office_revenue = intercept + coefficient * imdb_score

                st.markdown(f"### Predicted IMDB Score: **{imdb_score:.2f}**")
                st.markdown(f"### Predicted Box Office Revenue: **${box_office_revenue:,.2f}**")

if __name__ == '__main__':
    main()
