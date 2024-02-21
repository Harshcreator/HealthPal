from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
classifier = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-large-mnli"
)

@app.route('/classify', methods=['POST'])
def classify_entry():
    entry_text = request.json.get('entryText', '')
    if entry_text:
        # Perform zero-shot classification
        result = classifier(entry_text, ["positive", "negative", "neutral"], multi_class=True)
        return jsonify(result)
    else:
        return jsonify({'error': 'Entry text not provided'})

if __name__ == '__main__':
    app.run(debug=True)
