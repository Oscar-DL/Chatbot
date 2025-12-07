from flask import Flask, request, jsonify, render_template
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Evita problemas CPU/GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Fuerza CPU

lemmatizer = WordNetLemmatizer()

# Carga UTF-8 del intents
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Modelo DNN (solo CPU compatible)
model = load_model('chatbot_model.h5', compile=False)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Respuestas correctamente en español

# historial conversación
conversation_history = []


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    max_index = np.argmax(res)
    category = classes[max_index]
    return category


def get_response(tag, intents_json):
    for intent in intents_json['intents']:
        if intent["tag"] == tag:
            return random.choice(intent['responses'])
    return "No tengo respuesta para eso."


@app.route('/', methods=['GET', 'POST'])
def chat():
    global conversation_history
    if request.method == 'POST':
        user_message = request.form.get('message', '')
        if user_message:
            response = get_response(predict_class(user_message), intents)
            conversation_history.append({'user': user_message, 'bot': response})
    return render_template("index.html", history=conversation_history)


@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(force=True)
    user_message = data.get('message', '')
    if user_message:
        response = get_response(predict_class(user_message), intents)
        return jsonify({'response': response}), 200, {"Content-Type": "application/json; charset=utf-8"}

    return jsonify({'error': 'No message provided'}), 400


if __name__ == '__main__':
    app.run(debug=True)
