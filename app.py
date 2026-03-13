from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import re

app = Flask(__name__)
CORS(app)

HF_API_KEY = os.environ.get("HF_API_KEY", "")
HF_MODEL_URL = "https://api-inference.huggingface.co/models/Hello-SimpleAI/chatgpt-detector-roberta"

def detect_with_hf(text):
    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": text[:512]}
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=15)
        result = response.json()

        if isinstance(result, list) and len(result) > 0:
            scores = result[0]
            fake_score = 0
            real_score = 0
            for item in scores:
                if item['label'] == 'LABEL_0':
                    real_score = item['score']
                elif item['label'] == 'LABEL_1':
                    fake_score = item['score']

            ai_probability = round(fake_score * 100)
            trust_score = round(real_score * 100)
            return trust_score, ai_probability, True

    except Exception as e:
        print(f"HF API error: {e}")

    return None, None, False


def fallback_analyze(text):
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 4]

    transitions = [
        'furthermore', 'additionally', 'moreover', 'however',
        'consequently', 'nevertheless', 'therefore', 'in conclusion',
        'in summary', 'it is important', 'it should be noted',
        'as a result', 'on the other hand', 'notably', 'significantly'
    ]
    lc = text.lower()
    trans_count = sum(1 for t in transitions if t in lc)
    trans_score = min(trans_count * 8, 32)

    avg_len = len(words) / max(len(sentences), 1)
    sent_score = 30 if 14 < avg_len < 28 else 0

    human_punct = len(re.findall(r'[—–…\'\'""\(\)]', text))
    punct_score = 20 if human_punct < 2 else 0

    unique = set(w.lower().strip('.,!?') for w in words)
    richness = len(unique) / max(len(words), 1)
    rich_score = 10 if richness < 0.55 else 0

    starters = [' '.join(s.split()[:2]).lower() for s in sentences]
    unique_starters = len(set(starters)) / max(len(starters), 1)
    rep_score = 15 if unique_starters < 0.7 else 0

    ai_score = min(trans_score + sent_score + punct_score + rich_score + rep_score, 97)
    trust_score = max(100 - ai_score, 3)
    return trust_score, ai_score


def build_signals(trust_score, ai_probability, used_hf):
    signals = []
    if used_hf:
        if ai_probability > 70:
            signals.append({"type": "positive", "icon": "⚡", "title": "AI Model Detection — High Confidence", "desc": f"RoBERTa model detected strong AI patterns. AI probability: {ai_probability}%."})
        elif ai_probability > 40:
            signals.append({"type": "positive", "icon": "⚡", "title": "AI Model Detection — Mixed Signals", "desc": f"RoBERTa model found both AI and human patterns. AI probability: {ai_probability}%."})
        else:
            signals.append({"type": "negative", "icon": "✓", "title": "AI Model Detection — Human Likely", "desc": f"RoBERTa model found strong human writing patterns. AI probability: {ai_probability}%."})
        signals.append({"type": "negative" if trust_score > 60 else "positive", "icon": "🤖", "title": "Model: roberta-base-openai-detector", "desc": "Powered by OpenAI-trained RoBERTa detection model via Hugging Face."})
    else:
        signals.append({"type": "positive" if ai_probability > 50 else "negative", "icon": "⚡", "title": "Heuristic Analysis (API unavailable)", "desc": "Using pattern-based detection. HF API unavailable or model loading."})
    return signals


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    text = data['text'].strip()
    if len(text) < 30:
        return jsonify({"error": "Text too short. Minimum 30 characters."}), 400

    trust_score, ai_probability, used_hf = detect_with_hf(text)
    if trust_score is None:
        trust_score, ai_probability = fallback_analyze(text)
        used_hf = False

    signals = build_signals(trust_score, ai_probability, used_hf)
    return jsonify({"trust": trust_score, "ai_probability": ai_probability, "signals": signals, "model": "roberta-base-openai-detector" if used_hf else "heuristic-fallback"})


@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "LUMORA API running", "version": "0.2"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
