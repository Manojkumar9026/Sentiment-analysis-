# Sentiment-analysis-
Social Media Post Sentiment Analysis
This project classifies social media posts as Positive, Negative, or Neutral based on text content using a Naive Bayes classifier. The application is built using Python, with Flask for deployment, and makes use of libraries such as pandas, scikit-learn, and nltk for text preprocessing, feature extraction, and machine learning.

Table of Contents
Introduction
Features
Setup and Installation
Usage
Project Structure
Technologies Used
Model Deployment
License
Introduction
The goal of this project is to build a sentiment analysis model to classify social media posts. The dataset consists of various social media posts labeled as Positive, Negative, or Neutral. The project involves:

Text preprocessing (cleaning and lemmatizing text).
Feature extraction using TF-IDF.
Model training using a Naive Bayes classifier.
Model deployment via a Flask web application.
Features
Text Preprocessing: Removes numbers, special characters, and stop words, and applies lemmatization.
TF-IDF Feature Extraction: Converts text into numerical features.
Naive Bayes Model: Predicts the sentiment of posts.
Flask App: Deployed model accessible via a web interface.
Setup and Installation
Prerequisites
Ensure you have the following installed:

Python 3.8 or higher
pip (Python package installer)
Installation Steps
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/social-media-sentiment-analysis.git
cd social-media-sentiment-analysis
Install the required libraries:

bash
Copy code
pip install -r requirements.txt
Download NLTK data (if not already installed):

python
Copy code
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
Prepare the Dataset:

Place your dataset (social_media_post.csv) in the d:// directory or update the file_path in the code to point to the correct location.
Train the model: Run the train.py file (if provided), or use the script in your project to train the model and save the model and vectorizer as pickle files:

bash
Copy code
python app.py
Usage
Run the Flask App:

bash
Copy code
python app.py
Access the app: Open your browser and go to http://127.0.0.1:5000/.

Sentiment Prediction:

Enter the social media post in the text box.
Get sentiment prediction (Positive, Negative, or Neutral).
Project Structure
bash
Copy code
├── app.py              # Main Flask application
├── sentiment_model.pkl  # Saved Naive Bayes model
├── vectorizer.pkl       # Saved TF-IDF vectorizer
├── requirements.txt     # Python package dependencies
├── templates/
│   └── index.html       # HTML file for Flask front-end
└── social_media_post.csv # Dataset file
Technologies Used
Python: Core programming language.
pandas: Data handling and manipulation.
scikit-learn: Feature extraction and machine learning.
nltk: Natural language processing.
Flask: Web application framework.
HTML/CSS: Web interface for user input.
Model Deployment
The model and vectorizer are saved as pickle files and loaded into the Flask app to make real-time predictions on user input.
