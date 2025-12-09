from flask import Flask, request, jsonify, render_template
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from rapidfuzz import process

lemmatizer = WordNetLemmatizer()

#Importamos los archivos generados en el código anterior
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Enable UTF-8 encoding for JSON responses

# Global conversation history
conversation_history = []

#Pasamos las palabras de oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def best_match(user_text):
    # Tomamos todos los patrones del intents.json
    patterns = []
    for intent in intents["intents"]:
        patterns.extend(intent["patterns"])
    
    match, score, index = process.extractOne(user_text, patterns)
    return match if score > 70 else user_text

#Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

#Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res ==np.max(res))[0][0]
    category = classes[max_index]
    return category

#Obtenemos una respuesta aleatoria
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i["tag"]==tag:
            return random.choice(i['responses'])

@app.route('/', methods=['GET', 'POST'])
def chat():
    global conversation_history
    if request.method == 'POST':
        user_message = request.form.get('message', '')

        if user_message:
            fixed = best_match(user_message)
            response = get_response(predict_class(fixed), intents)
            conversation_history.append({'user': user_message, 'bot': response})
    return render_template("index.html", history=conversation_history)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    user_message = data.get('message', '')
    if user_message:
        fixed = best_match(user_message)
        response = get_response(predict_class(fixed), intents)
        return jsonify({'response': response})
    return jsonify({'error': 'No message provided'}), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



