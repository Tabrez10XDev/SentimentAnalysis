from tensorflow import keras
import tensorflow as tf
import re
from keras.utils import pad_sequences
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
nltk.download('punkt')

filename = 'fin_model2.h5'
model = keras.models.load_model(filename)
class_names = ["anger","sadness","fear","joy","surprise","love"]
tokenizer = Tokenizer()
f=open('train.txt','r')
x_train=[]
y_train=[]
for i in f:
    l=i.split(';')
    y_train.append(l[1].strip())
    x_train.append(l[0])
f=open('test.txt','r')
x_test=[]
y_test=[]
x_real_test = []
y_real_test = []
for i in f:
    l=i.split(';')
    y_real_test.append(l[1].strip())
    x_real_test.append(l[0])
f=open('val.txt','r')
for i in f:
    l=i.split(';')
    y_test.append(l[1].strip())
    x_test.append(l[0])
data_train=pd.DataFrame({'Text':x_train,'Emotion':y_train})
data_test=pd.DataFrame({'Text':x_test,'Emotion':y_test})
data_real_test= pd.DataFrame({'Text':x_real_test,'Emotion':y_real_test})
data_train.append(data_real_test,ignore_index=True)
data=data_train.append(data_test,ignore_index=True)

def clean_text(data):
    data=re.sub(r"(#[\d\w\.]+)", '', data)
    data=re.sub(r"(@[\d\w\.]+)", '', data)
    data=word_tokenize(data)
    return data
texts=[' '.join(clean_text(text)) for text in data.Text]
texts_train=[' '.join(clean_text(text)) for text in x_train]
texts_test=[' '.join(clean_text(text)) for text in x_test]
texts_real_test=[' '.join(clean_text(text)) for text in x_real_test]
tokenizer.fit_on_texts(texts)




def features():
    return "Hello World"

def diagnose(lis):
    print(lis)
    message = lis
    seq = tokenizer.texts_to_sequences(message)
    padded = pad_sequences(seq, maxlen=500)
    pred = model.predict(padded)
    return class_names[np.argmax(pred)]



