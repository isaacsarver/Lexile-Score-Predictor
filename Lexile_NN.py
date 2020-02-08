# Load required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import time
import pickle
import seaborn as sns
import sys
import random
from sklearn.preprocessing import LabelEncoder
sys.setrecursionlimit(1500)
get_ipython().run_line_magic('matplotlib', 'inline')

import keras

from keras.layers import Input, Dense
from keras.models import Model,load_model
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard


from sklearn.preprocessing import LabelEncoder


# Luke's Preprocess function

def Preprocess(data, test_size):
    #data is in the format: [filename]
    #X = []
    #y = []
    Xy = []
    lexile_converter = [700,800,900,1000,1100,1200,1300]#7 bins, value is the max of that bin
    for filename in data:
        directory = "/Users/home/CSE 842 Project NN/Book Files/" + filename
        fp = open(directory,'r', encoding="utf8")
        first_line = fp.readline().strip()#lexile measure is in this line
        
        index = 0
        for i, ch in enumerate(first_line):#this shears off weird formatting characters off line 1
            if ch.isdigit():
                index = i
                break
        lexile = int(first_line[index:])
        lexile_bin = 1300
        for value in lexile_converter:#find what bin our lexile measure falls into
            if lexile <= value:
                lexile_bin = value
                break
        contents = ""
        count = 0
        for line in fp:#add 100 word chunks to be used as data point
            line = line.strip()
            contents = contents + line
            count+=1
            if count==100:
                count=0
                #X.append(contents)
                #y.append(lexile_bin)
                Xy.append((contents,lexile_bin))
                contents = ""
    
    #shuffle up our dataset
    random.shuffle(Xy)
    
    """
    X_train = X[:int(len(X)*(1-test_size))]
    X_test = X[int(len(X)*(1-test_size)):]
    
    y_train = y[:int(len(y)*(1-test_size))]
    y_test = y[int(len(y)*(1-test_size)):]
    """
    Xy_train = Xy[:int(len(Xy)*(1-test_size))]
    Xy_test = Xy[int(len(Xy)*(1-test_size)):]
    
    X_train = [item[0] for item in Xy_train]
    y_train = [item[1] for item in Xy_train]
    
    X_test = [item[0] for item in Xy_test]
    y_test = [item[1] for item in Xy_test]
    
    
    
    return X_train, y_train, X_test, y_test


# Get filenames (this was different for me, I only had 34)

data = []
for i in range(1, 35):
    #filename = "./Book Files/" + str(i) + ".txt"
    filename = str(i) + ".txt"
    data.append(filename)


# Preprocess

X_train,y_train,X_test,y_test = Preprocess(data,1/5)
print(len(X_train))
#print(X_train[0])


# Create train and test dataframes

train = pd.DataFrame(list(zip(X_train, y_train)), columns = ['Text', 'Score'])
test = pd.DataFrame(list(zip(X_test, y_test)), columns = ['Text', 'Score'])


# In[7]:

def token_count(text):
    'function to count number of tokens'
    length=len(text.split())
    return length

def tokenize(text):
    "tokenize the text using default space tokenizer"
    lines=(line for line in text.split("\n") )
    tokenized=""
    for sentence in lines:
        tokenized+= " ".join(tok for tok in sentence.split())
    return tokenized


# Apply tokenize and token count

train['tokenized_text'] = train['Text'].apply(tokenize)
train['token_count'] = train['tokenized_text'].apply(token_count)

test['tokenized_text'] = test['Text'].apply(tokenize)
test['token_count'] = test['tokenized_text'].apply(token_count)


# Concatenate dataframes

data = pd.concat([train, test])


# Create TF-IDF for the text

num_max = 4000

def train_tf_idf_model(texts):
    "train tf idf model"
    tok = Tokenizer(num_words=num_max)
    tok.fit_on_texts(texts)
    return tok

def prepare_model_input(tfidf_model, dataframe, mode='tfidf'):
    "function to prepare data input features using tfidf model"
    le = LabelEncoder()
    sample_texts = list(dataframe['tokenized_text'])
    sample_texts = [' '.join(x.split()) for x in sample_texts]
    
    targets=list(dataframe['Score'])
    sample_target = le.fit_transform(targets)
    
    if mode=='tfidf':
        sample_texts=tfidf_model.texts_to_matrix(sample_texts,mode='tfidf')
    else:
        sample_texts=tfidf_model.texts_to_matrix(sample_texts)
    
    print('shape of labels: ', sample_target.shape)
    print('shape of data: ', sample_texts.shape)
    
    return sample_texts, sample_target

# Train TF-IDF
texts = list(data['tokenized_text'])
tfidf_model = train_tf_idf_model(texts)
# prepare model input data
mat_texts, tags = prepare_model_input(tfidf_model, data, mode='tfidf')


# Split into training and test data

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(mat_texts, tags, test_size=0.15)
print ('train data shape: ', X_train.shape, y_train.shape)
print ('validation data shape :' , X_val.shape, y_val.shape)

## Define and initialize the network

model_save_path="checkpoints/Lexile_NN.h5"


# Model architecture

def get_simple_model():
    """
    Uses 3 layers: Input -> L1 : (Linear -> Relu) -> L2: (Linear -> Relu)-> (Linear -> Sigmoid)
    Layer L1 has 512 neurons with Relu activation
    Layer L2 has 256 neurons with Relu activation
    Regularization : We use dropout with probability 0.5 for L1, L2 to prevent overfitting
    Loss Function : binary cross entropy. Don't know why, but this works best.
    Optimizer : We use Adam optimizer for gradient descent estimation (faster optimization)
    Data Shuffling : Data shuffling is set to true
    Batch Size : 64
    Learning Rate = 0.001
    """
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(num_max,)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc',keras.metrics.categorical_accuracy])
    print('compile done')
    return model

def check_model(model,x,y,epochs=2):
    history=model.fit(x,y,batch_size=32,epochs=epochs,verbose=1,shuffle=True,validation_split=0.2,
              callbacks=[checkpointer, tensorboard]).history
    return history


def check_model2(model,x_train,y_train,epochs=10):
    history=model.fit(x_train,y_train,batch_size=64,
                      epochs=epochs,verbose=1,
                      shuffle=True,
                      validation_split = 1/9,
                      callbacks=[checkpointer, tensorboard]).history
    return history

# define checkpointer
checkpointer = ModelCheckpoint(filepath=model_save_path,
                               verbose=1,
                               save_best_only=True)    

# define tensorboard
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

# define the predict function for the deep learning model for later use
def predict(data):
    result=spam_model_dl.predict(data)
    prediction = [round(x[0]) for x in result]
    return prediction

y_train = keras.utils.to_categorical(y_train)

y_val = keras.utils.to_categorical(y_val)


# Training
# get the compiled model
model = get_simple_model()

#make folder for checkpoints
# path = "Checkpoints/"
# os.mkdir(path)

# load history
# history=check_model(m,mat_texts,tags,epochs=10)
history=check_model2(model,X_train,y_train,epochs=10)

# Evaluate
scores = model.evaluate(X_val,y_val)

print(scores)

print(model.metrics_names)


