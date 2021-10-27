import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve

class IrisDataset(list):
    def __init__(self, controlled=True):
        super()
        irisData = load_iris()
        self.all_flowers = np.column_stack((irisData.data, irisData.target.T))
        self.training, self.test = self.controlled_shuffle() if controlled else self.shuffle()
        
    def shuffle(self):
        random.shuffle(self.all_flowers)
        training_size = (len(self.all_flowers) // 4) * 3
        training = self.all_flowers[:training_size]
        test = self.all_flowers[training_size:]
        return training, test
    
    def controlled_shuffle(self):
        flowers_per_type = int(len(self.all_flowers) / 3)
        training_per_type = int(flowers_per_type * 0.75)
        training = []
        test = []
        for i in range(3):
            start = i * flowers_per_type
            end = start + flowers_per_type            
            type_of_flowers = list(self.all_flowers[start:end])
            training += type_of_flowers[:training_per_type]
            test += type_of_flowers[training_per_type:]
        return training, test


def run_mlp(neurons):
    print("\nTesting MLP with %s neurons" % neurons)
    mlp = MLPClassifier(hidden_layer_sizes=(neurons,), max_iter=10000)
    mlp.fit(train_data, train_labels)
    
    predictions_train = mlp.predict(train_data)
    print("Training accuracy: %s" % accuracy_score(predictions_train, train_labels))
    predictions_test = mlp.predict(test_data)
    print("Prediction accuracy: %s" % accuracy_score(predictions_test, test_labels))
    
    train_sizes, train_scores, validation_scores = learning_curve(
    estimator=mlp,
    X=train_data,
    y=train_labels,
    scoring='neg_mean_squared_error')

    train_scores_mean = -train_scores.mean(axis = 1)
    validation_scores_mean = -validation_scores.mean(axis = 1)
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('MLP with %s neurons' % neurons, fontsize = 18, y = 1.03)
    plt.legend()
    plt.show()


dataset = IrisDataset()
train_data = np.array([i[:4] for i in dataset.training])
train_labels = np.array([i[4] for i in dataset.training])
test_data = np.array([i[:4] for i in dataset.test])
test_labels = np.array([i[4] for i in dataset.test])

run_mlp(1)
run_mlp(3)
run_mlp(5)
run_mlp(10)
