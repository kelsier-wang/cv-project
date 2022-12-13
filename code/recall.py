#Truman
#TODO:
# constructing csv or txt file with recalled labels
# write method that matches pill labels to recalled ones (efficiently) 
# think about closeness of string matching
# output boolean
import pandas as pd

data = None

def loaddata():
    data = pd.read_csv('../data/recall')
    return data

def recall(prediction):
    return data[' Recalled'][prediction]
