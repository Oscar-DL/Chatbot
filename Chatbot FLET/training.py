import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from rapidfuzz import process  # <<< NUEVO

#Para crear la red neuronal
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

#Clasifica los patrones y las categorías
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


# <<< NUEVO: función para corregir palabras
def correct_word(word, vocab, threshold=80):
    best = process.extractOne(word, vocab, score_cutoff=threshold)
    if best:
        return best[0]
    return word


#Pasa la información a unos y ceros según las palabras presentes en cada categoría
training = []
output_empty = [0]*len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]

    # <<< NUEVO: corrección fuzzy + lematización
    word_patterns = [
        correct_word(lemmatizer.lemmatize(word.lower()), words)
        for word in word_patterns
    ]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

print(len(training))

train_x=[]
train_y=[]
for i in training:
    train_x.append(i[0])
    train_y.append(i[1])

train_x = np.array(train_x)
train_y = np.array(train_y)


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), name="inp_layer", activation='relu'))
model.add(Dropout(0.3, name="hidden_layer1"))
model.add(Dense(64, name="hidden_layer2", activation='relu'))
model.add(Dropout(0.3, name="hidden_layer3"))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(train_y[0]), name="output_layer", activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=2500, batch_size=8, verbose=1)
model.save("chatbot_model.h5")
print("Modelo creado")
