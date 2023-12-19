from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open("model.pkl", "rb") as file:
    classifier = pickle.load(file)

# Load the data
df = pd.read_csv('lemma.csv', delimiter=",")

# Set up the TF-IDF vectorizer
tfidf = TfidfVectorizer(max_df=0.5, min_df=2)
tfidf.fit(df['lemma'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form.get['text','']

        # Vectorize the input text
        text_vectorized = tfidf.transform([text])

        # Make prediction using the pre-trained model
        prediction = classifier.predict(text_vectorized)

        # Map class labels to the desired values
        result_label = {
            0: 'Netral',
            1: 'Negatif',
            2: 'Positif'
        }

        # Pass the mapped prediction result to the result.html template
        return render_template('result.html', prediction=result_label[prediction[0]])

if __name__ == '__main__':
    app.run(debug=True)

print(request.form)
