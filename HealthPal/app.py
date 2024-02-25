from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis")

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data['text']
    result = sentiment_analysis(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
