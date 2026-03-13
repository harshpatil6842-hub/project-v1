from flask import Flask, request, jsonify
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

def analyze_text(text):
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 4]

    # Signal 1: Transition words
    transitions = [
        'furthermore', 'additionally', 'moreover', 'however',
        'consequently', 'nevertheless', 'therefore', 'in conclusion',
        'in summary', 'it is important', 'it should be noted',
        'as a result', 'on the other hand', 'notably', 'significantly'
    ]
    lc = text.lower()
    trans_count = sum(1 for t in transitions if t in lc)
    trans_score = min(trans_count * 8, 32)

    # Signal 2: Avg sentence length
    avg_len = len(words) / max(len(sentences), 1)
    sent_score = 30 if 14 < avg_len < 28 else 0

    # Signal 3: Punctuation variety
    human_punct = len(re.findall(r'[—–…\'\'""\(\)]', text))
    punct_score = 20 if human_punct < 2 else 0

    # Signal 4: Vocabulary richness
    unique = set(w.lower().strip('.,!?') for w in words)
    richness = len(unique) / max(len(words), 1)
    rich_score = 10 if richness < 0.55 else 0

    # Signal 5: Repetitive sentence starters
    starters = [' '.join(s.split()[:2]).lower() for s in sentences]
    unique_starters = len(set(starters)) / max(len(starters), 1)
    rep_score = 15 if unique_starters < 0.7 else 0

    ai_score = min(trans_score + sent_score + punct_score + rich_score + rep_score, 97)
    trust_score = max(100 - ai_score, 3)

    # Build signals
    signals = []

    if trans_count >= 2:
        signals.append({
            "type": "positive",
            "icon": "⚡",
            "title": "High transition word density",
            "desc": f"{trans_count} AI-typical phrase(s) detected (furthermore, moreover, etc.)"
        })
    else:
        signals.append({
            "type": "negative",
            "icon": "✓",
            "title": "Low AI transition phrase usage",
            "desc": "Writer avoids formulaic connectors — human-typical."
        })

    if 14 < avg_len < 28:
        signals.append({
            "type": "positive",
            "icon": "⚡",
            "title": "Uniform sentence length",
            "desc": f"~{round(avg_len)} words/sentence — consistent with AI generation."
        })
    else:
        signals.append({
            "type": "negative",
            "icon": "✓",
            "title": "Variable sentence rhythm",
            "desc": "Sentence length varies — more human-like flow."
        })

    if human_punct < 2:
        signals.append({
            "type": "positive",
            "icon": "⚡",
            "title": "Minimal punctuation variety",
            "desc": "Lacks em dashes, ellipses — typical of AI output."
        })
    else:
        signals.append({
            "type": "negative",
            "icon": "✓",
            "title": "Rich punctuation variety",
            "desc": "Contains stylistic punctuation — human writing marker."
        })

    if richness < 0.55:
        signals.append({
            "type": "positive",
            "icon": "⚡",
            "title": "Vocabulary repetition detected",
            "desc": f"{round(richness * 100)}% unique word ratio — below human average."
        })
    else:
        signals.append({
            "type": "negative",
            "icon": "✓",
            "title": "Strong vocabulary diversity",
            "desc": f"{round(richness * 100)}% unique word ratio — authentic writing signal."
        })

    return {
        "trust": trust_score,
        "ai_probability": ai_score,
        "signals": signals
    }


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text'].strip()

    if len(text) < 30:
        return jsonify({"error": "Text too short. Minimum 30 characters."}), 400

    result = analyze_text(text)
    return jsonify(result)


@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "LUMORA API running", "version": "0.1"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)