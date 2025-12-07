from flask import Flask, request, jsonify, render_template
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import os

# ---- CPU ONLY (force disable GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ---- NLTK download (safe for server)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

# ---- Load intents JSON UTF-8
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# ---- Load training artifacts
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# ---- Load TensorFlow model CPU-friendly
model = load_model('chatbot_model.h5', compile=False)

# ---- Flask
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Proper UTF-8 output

conversation_history = []


# ===========================
# NLP FUNCTIONS
# ===========================

def clean_up_sentence(sentence: str):
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence: str):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence: str):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    max_index = np.argmax(res)
    return classes[max_index]


def get_response(tag: str, intents_json: dict):
    for intent in intents_json['intents']:
        if intent["tag"] == tag:
            return random.choice(intent['responses'])
    return "No tengo respuesta para eso."


# ===========================
# ROUTES
# ===========================

@app.route('/', methods=['GET', 'POST'])
def chat():
    global conversation_history

    if request.method == 'POST':
        user_message = request.form.get('message', '')
        if user_message:
            response = get_response(predict_class(user_message), intents)
            conversation_history.append({
                'user': user_message,
                'bot': response
            })

    return render_template("index.html", history=conversation_history)


@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(force=True)
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    response = get_response(predict_class(user_message), intents)

    return (
        jsonify({'response': response}),
        200,
        {"Content-Type": "application/json; charset=utf-8"}
    )


# ===========================
# RUN
# ===========================

if __name__ == '__main__':
    app.run(debug=True)
