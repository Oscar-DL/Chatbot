from flask import Flask, render_template, request, jsonify
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(__file__)
FAQ_PATH = os.path.join(BASE_DIR, 'faqs.json')

def load_faqs(path=FAQ_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

faqs = load_faqs()
corpus = [f"{item['question']} {item['answer']}" for item in faqs]

# Carga del modelo de sentence-transformers. Puedes cambiar el modelo a través de la variable
# de entorno `ST_MODEL`. El modelo se descargará la primera vez que se use.
MODEL_NAME = os.environ.get('ST_MODEL', 'all-MiniLM-L6-v2')
model = None
corpus_embeddings = None
if corpus:
    try:
        model = SentenceTransformer(MODEL_NAME)
        corpus_embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
    except Exception:
        # Si falla la carga del modelo, dejaremos corpus_embeddings como None
        model = None
        corpus_embeddings = None

def find_best_answer(question_text, threshold=0.45):
    if model is None or corpus_embeddings is None:
        return {'answer': "Lo siento, el modelo de embeddings no está disponible.", 'score': 0.0, 'matched_question': None}

    q_emb = model.encode([question_text], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, corpus_embeddings).flatten()
    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])
    if best_score < threshold:
        return {'answer': "No encontré una respuesta clara en las FAQs. Intenta reformular la pregunta.", 'score': best_score, 'matched_question': None}
    return {'answer': faqs[best_idx]['answer'], 'score': best_score, 'matched_question': faqs[best_idx]['question']}

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def api_ask():
    data = request.get_json(force=True)
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'Pregunta vacía.'}), 400
    result = find_best_answer(question)
    return jsonify({'question': question, 'answer': result['answer'], 'score': result['score'], 'matched_question': result['matched_question']})

if __name__ == '__main__':
    app.run(debug=True)
