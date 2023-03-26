import nltk
import json
import pickle
import random
import numpy as np

ignore_words = ["?", "!", ".", ",", "'m", "'s"]
import tensorflow
from data_preprocessing import get_stem_words

model = tensorflow.keras.models.load_model("./chatbot_model.h5")
intents = json.loads(open("./intents.json").read())

words = pickle.load(open("./words.pkl","rb"))

classes = pickle.load(open("./classes.pkl", "rb"))

def preprocess_userinput(userinput):
    inputwordToken1 = nltk.word_tokenize(userinput)
    inputwordToken2 = get_stem_words(inputwordToken1, ignore_words)
    inputwordToken2 = sorted(list(set(inputwordToken2)))
    bag_of_words = []
    bag = []
    for word in words:
        if word in inputwordToken2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    
    bag.append(bag_of_words)
    return np.array(bag)


def botClassPrediction(userinput):
    input = preprocess_userinput(userinput)
    prediction = model.predict(input)
    predict_classLabel = np.argmax(prediction[0])
    return predict_classLabel


def botResponse(userinput):
    predict_classLabel = botClassPrediction(userinput)
    predicted_class = classes[predict_classLabel]
    for intent in intents["intents"]:
        if intent["tag"] == predicted_class:
            botResponse = random.choice(intent["responses"])
    return botResponse



