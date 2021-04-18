import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from StarsDataset import DatasetPanda

MAX_K = 50
TRIES_FOR_EACH_K = 500

def new_dataset():
    return DatasetPanda(drop_columns=["Color", "Spectral_Class"])

def run_knn():
    data = new_dataset()

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(data.scaled_train, data.train_classes)
    test_pred = classifier.predict(data.scaled_test)
    
    print(confusion_matrix(data.test_classes, test_pred))
    print(classification_report(data.test_classes, test_pred))
    
def get_knn_error(k, data):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data.scaled_train, data.train_classes)
    pred_i = knn.predict(data.scaled_test)
    return np.mean(pred_i != data.test_classes)

def plot_knn_average_error():
    total_error = []
    for i in range(1, MAX_K):
        error = []
        for j in range(0, TRIES_FOR_EACH_K):
            error.append(get_knn_error(i, new_dataset()))
        total_error.append(sum(error) / len(error))
        
    plt.figure(figsize=(24, 12))
    plt.plot(range(1, MAX_K), 
             total_error, 
             color='red', 
             linestyle='dashed', 
             marker='o',
             markerfacecolor='black', 
             markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')

if __name__ == '__main__':
    run_knn()
    plot_knn_average_error()
    