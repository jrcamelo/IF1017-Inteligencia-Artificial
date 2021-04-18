import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FILENAME = "Stars.csv"
TRAIN_TO_TEST_RATIO = 3/4

CLASSIFICATION = {
    0: "Red Dwarf",
    1: "Brown Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Super Giants",
    5: "Hyper Giants",
    None: "?"
}

COL = {
    "Temperature": 0,
    "Luminosity": 1,
    "Radius": 2,
    "Magnitude": 3,
    "Color": 4,
    "SMASS": 5,
    "Classification": 6        
}

class Dataset(object):
    def __init__(self):
        pass
    
    def drop_if_necessary(self, drop_columns):
        if drop_columns:
            self.original = self.original.drop(columns=drop_columns)

class DatasetNumpy(Dataset):
    def __init__(self, drop_columns=None):
        self.original = pd.read_csv(FILENAME)
        self.drop_if_necessary(drop_columns)
        self.original = self.original.to_numpy()
        self.train, self.test, self.train_classes, self.test_classes = train_test_split(
            np.array([i[:COL["Classification"]] for i in self.original]), 
            np.array([i[COL["Classification"]] for i in self.original]), 
            train_size=TRAIN_TO_TEST_RATIO, 
            stratify=np.array([i[COL["Classification"]] for i in self.original]))

        

class DatasetPanda(Dataset):    
    def __init__(self, drop_columns=None):
        self.original = pd.read_csv(FILENAME)
        self.drop_if_necessary(drop_columns)
        self.train, self.test, self.train_classes, self.test_classes = train_test_split(
            self.original, 
            self.original["Type"], 
            train_size=TRAIN_TO_TEST_RATIO, 
            stratify=self.original["Type"])
        scaler = StandardScaler()
        scaler.fit(self.train)
        self.scaled_train = scaler.transform(self.train)
        self.scaled_test = scaler.transform(self.test)
        
        
def print_star(star, star_classification=None):
    print("Star - Temp: {0}, Lum: {1}, Rad: {2}, Magnitude: {3}, Color: {4}, SMASS: {5}, Class: {6}".format(
        star[0], truncate(star[1], 5), truncate(star[2], 5), truncate(star[3], 2), star[4], star[5], CLASSIFICATION[star_classification]))

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

