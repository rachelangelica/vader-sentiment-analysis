from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('vader_lexicon')
app = Flask(__name__)
factory = StemmerFactory()
stemmer = factory.create_stemmer()



@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        inp = request.form.get("inp")
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(inp)
        if score['compound'] >= 0.05:
            sentiment = "Positive 😀"
        elif score['compound'] <= -0.05:
            sentiment = "Negative 😡"
        else:
            sentiment = "Neutral 😐"
        return render_template('home.html', sentiment=sentiment)
    return render_template('home.html')

if __name__ == "__main__":
    app.run()
