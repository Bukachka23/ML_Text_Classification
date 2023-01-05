import sys
import numpy as np
import pickle
import re
from Stemmer import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def text_cleaner(text):                                                                # Defines a function called "text_cleaner" that takes in a variable called "text".
    text = text.lower()                                                                # Converts the text to all lowercase.
    stemmer = Stemmer('english')                                                       # Creates a stemmer object for Russian language text.
    text = ' '.join( stemmer.stemWords( text.split() ) )                               # Stems the words in the text and joins them with spaces, then assigns the result to the "text" variable.
    text = re.sub( r'\b\d+\b', ' digit ', text )                                       # Substitutes any sequence of digits with the string "digit".
    return  text                                                                       # Returns the modified "text" variable.


def load_data():                                                                       # Defines a function called "load_data" that does not take any arguments.
    data = {'text':[],'tag':[]}                                                        # Initializes an empty dictionary called "data" with keys "text" and "tag", and empty lists as values.
    for line in open('data_1.txt'):                                                    # Opens the file "data_1.txt" and iterates over each line in the file.
        if not('#' in line):                                                           # If the line does not contain the "#" character, the line is split at the "@" character and the resulting elements are added to the "text" and "tag" lists in the "data" dictionary.
            row = line.split("@")
            data['text'] += [row[0]]
            data['tag'] += [row[1]]
    return data                                                                        # Returns the "data" dictionary.

def train_test_split(data, validation_split = 0.1):                                    # Defines a function called "train_test_split" that takes in two arguments, "data" and "validation_split". The latter has a default value of 0.1.                                   #
    sz = len(data['text'])                                                             # Calculates the length of the "text" list in the "data" dictionary and assigns it to a variable "sz".
    indices = np.arange(sz)                                                            # Creates an array of indices ranging from 0 to "sz" - 1.
    np.random.shuffle(indices)                                                         # Shuffles the indices randomly.
    X = [data['text'][i] for i in indices]                                             # Creates a new list "X" consisting of the elements in the "text" list from the "data" dictionary at the shuffled indices.
    Y = [data['tag'][i] for i in indices]                                              # Creates a new list "Y" consisting of the elements in the "tag" list from the "data" dictionary at the shuffled indices.
    nb_validation_samples = int( validation_split * sz )                               # Calculates the number of validation samples as the product of "validation_split" and "sz", and rounds it down to the nearest integer.

    return {                                                                           # Returns a dictionary with two keys, "train" and "test". The values are dictionaries with keys "x" and "y" and values corresponding to the training and testing sets of "X" and "Y", respectively. The testing set consists of the last "nb_validation_samples" elements of "X" and "Y"
        'train': {'x': X[:-nb_validation_samples], 'y': Y[:-nb_validation_samples]},
        'test': {'x': X[-nb_validation_samples:], 'y': Y[-nb_validation_samples:]}
    }


def openai():                                                                          # Defines a function called "openai" that does not take any arguments.
    data = load_data()                                                                 # Calls the "load_data" function and assigns the returned dictionary to a variable "data".
    D = train_test_split(data)                                                         # Calls the "train_test_split" function with "data" as an argument and assigns the returned dictionary to a variable "D".
    text_clf = Pipeline([                                                              # Initializes a pipeline object with two steps, "tfidf" and "clf". The "tfidf" step applies the TfidfVectorizer method to the input data, and the "clf" step applies the SGDClassifier method with the "hinge" loss function.
                    ('tfidf', TfidfVectorizer()),
                    ('clf', SGDClassifier(loss='hinge')),
                    ])
    text_clf.fit(D['train']['x'], D['train']['y'])                                     # Fits the pipeline object to the training data in the "D" dictionary.
    predicted = text_clf.predict( D['train']['x'] )                                    # Makes predictions on the training data using the fitted pipeline object and assigns the result to a variable "predicted".
    z = input("Enter a question without a question mark at the end: ")                 # Prompts the user to input a question without a question mark at the end.
    zz = []                                                                            # Initializes an empty list called "zz".
    zz.append(z)                                                                       # Appends the user-inputted question to the "zz" list.
    predicted = text_clf.predict(zz)                                                   # Makes predictions on the "zz" list using the fitted pipeline object and assigns the result to the "predicted" variable.
    print(predicted[0])                                                                # Prints the first element of the "predicted" list.


if __name__ == '__main__':
    sys.exit(openai())