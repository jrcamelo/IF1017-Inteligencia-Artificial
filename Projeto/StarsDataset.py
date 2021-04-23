import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


FILENAME = "Stars.csv"
TRAIN_TO_TEST_RATIO = 3/4

    
CLASSIFICATION = [
    "Red Dwarf",
    "Brown Dwarf",
    "White Dwarf",
    "Main Sequence",
    "Super Giants",
    "Hyper Giants",
]

class Dataset(object):    
    def __init__(self):
        pass
        
    def label_to_index(self, label):
        return self.LABELS.index(label)
    
    def column_to_index(self, column):
        return self.COLUMNS.index(column)
        
    def label_to_column(self, label):
        return self.COLUMNS[self.label_to_index(label)]
        
    def print_star(self, star, star_classification=None):
        print("Star - Temp: {0}, Lum: {1}, Rad: {2}, Magnitude: {3}, Color: {4}, SMASS: {5}, Class: {6}".format(
            star[0], 
            self.truncate(star[1], 5), 
            self.truncate(star[2], 5), 
            self.truncate(star[3], 2), 
            star[4], 
            star[5], 
            CLASSIFICATION[star_classification]))
    
    def truncate(self, n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier


class DatasetPanda(Dataset):
    def __init__(self, target="Classification"):
        self.original = pd.read_csv(FILENAME)
        self.setup_lists()
        self.normalize_color()
        self.define_target(target)
        self.set_target_labels()
        self.flatten_data()
        self.set_target()
        self.split()
        self.scale()
    
    # Lists end up Class variables instead of Instance variables if not done here
    def setup_lists(self):
        # Names as they should appear in the plots
        self.LABELS = [
            "Temperature",
            "Luminosity",
            "Radius",
            "Magnitude",
            "Color",
            "SMASS",
            "Classification",
        ]        
        # Names as they are in the data
        self.COLUMNS = [
            "Temperature",
            "L",
            "R",
            "A_M",
            "Color",
            "Spectral_Class",
            "Type",
        ]        
        
    # Color is really unorganized, this is a bit unfair but helps
    def normalize_color(self):
        col = "Color"        
        self.original[col] = self.original[col].str.lower()
        self.original[col] = self.original[col].str.replace(" ", "-")
        self.original[col] = self.original[col].str.replace("whitish", "white")
        self.original[col] = self.original[col].str.replace("ish", "")
        self.original[col] = self.original[col].str.replace("orange-red", "orange")
        self.original[col] = self.original[col].str.replace("pale-yellow-orange", "orange")
        self.original[col] = self.original[col].str.replace("white-yellow", "yellow")
    
    # Sets the index and column name
    def define_target(self, target_label):
        self.target_index = self.label_to_index(target_label)
        self.target_column = self.label_to_column(target_label)
    
    # Gets the names before text is transformed into numbers
    def set_target_labels(self):
        if self.target_column == "Type":
            self.target_classes = CLASSIFICATION
            return
        target_values = self.original[self.target_column]
        self.target_classes = list(set(target_values))
    
    # Changes text into numbers
    def flatten_data(self):
        label_encoder = LabelEncoder()
        text_columns = ["Color", "Spectral_Class"]
        flattened_text = self.original[text_columns].values.flatten()
        label_encoder.fit(flattened_text)
        flattened_columns = label_encoder.fit_transform
        self.original[text_columns] = self.original[text_columns].apply(flattened_columns)
    
    # Gets the target values and drops it from the data
    def set_target(self):
        self.target = self.original[self.target_column]
        self.drop_column(self.target_column)
    
    # Removes from the data, then the lists
    def drop_labels(self, labels=None):
        if labels:
            for label in labels:
                column = self.label_to_column(label)
                self.original = self.original.drop(columns=column)
            self.drop_labels_from_lists(labels)
    
    # Same as above, but "private"
    def drop_column(self, column):
        self.original = self.original.drop(columns=column)
        self.LABELS.pop(self.column_to_index(column))
        self.COLUMNS.pop(self.column_to_index(column))
    
    # Removes from the list of columns and labels
    def drop_labels_from_lists(self, labels):
        for label in labels:
            self.COLUMNS.pop(self.label_to_index(label))
            self.LABELS.pop(self.label_to_index(label))
        
    # Splits into test and training in a balanced manner
    def split(self):
        self.train, self.test, self.train_classes, self.test_classes = train_test_split(
            self.original, 
            self.target, 
            train_size=TRAIN_TO_TEST_RATIO, 
            stratify=self.target)        
        
    # "Standardizes features by removing the mean and scaling to unit variance"
    # Helps on simpler comparations
    def scale(self):
        scaler = StandardScaler()
        scaler.fit(self.train)
        self.scaled_train = scaler.transform(self.train)
        self.scaled_test = scaler.transform(self.test)
        
        

