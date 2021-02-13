import re
import random
import math 
from enum import Enum
import copy 


class Iris(Enum):
    def __str__(self):
        return str(self.value)
    
    SETOSA = "Iris-setosa"
    VERSICOLOR = "Iris-versicolor"
    VIRGINICA = "Iris-virginica"



class Flower:
    def __init__(self, 
                 sepalLength, 
                 sepalWidth, 
                 petalLength, 
                 petalWidth, 
                 classification, 
                 guess=None,
                 distance=None):
        self.sepalLength = float(sepalLength)
        self.sepalWidth = float(sepalWidth)
        self.petalLength = float(petalLength)
        self.petalWidth = float(petalWidth)
        self.classification = classification
        self.guess = guess
        
    def calculate_distance(self, x, y):
        return math.sqrt((x ** 2) - (2 * x * y) + (y ** 2))
        
    def sepal_length_distance(self, other):
        return self.calculate_distance(self.sepalLength, other.sepalLength)
    
    def sepal_width_distance(self,other):
        return self.calculate_distance(self.sepalWidth, other.sepalWidth)
    
    def petal_length_distance(self,other):
        return self.calculate_distance(self.petalLength, other.petalLength)
    
    def petal_width_distance(self,other):
        return self.calculate_distance(self.petalWidth, other.petalWidth)
    
    def set_distance_to(self,other):
        self.distance_to_current_flower = (
            self.sepal_length_distance(other) + 
            self.sepal_width_distance(other) +
            self.petal_length_distance(other) +
            self.petal_width_distance(other)
        )
        
    def make_guess(self, closeFlowers):
        setosa, versicolor, virginica = self.count_types_of_close_flowers(closeFlowers)
        if setosa > versicolor and setosa > virginica: 
            self.guess = str(Iris.SETOSA)
        elif versicolor > setosa and versicolor > virginica:
            self.guess = str(Iris.VERSICOLOR)
        elif virginica > setosa and virginica > versicolor:
            self.guess = str(Iris.VIRGINICA)
        else:
            self.guess = "Uknown"
    
    def count_types_of_close_flowers(self, closeFlowers):        
        setosa = self.count_close_flowers_of_type(str(Iris.SETOSA), closeFlowers)
        versicolor = self.count_close_flowers_of_type(str(Iris.VERSICOLOR), closeFlowers)
        virginica = self.count_close_flowers_of_type(str(Iris.VIRGINICA), closeFlowers)
        return setosa, versicolor, virginica
        
    def count_close_flowers_of_type(self, flowerType, closeFlowers):
        return len([flower for flower in closeFlowers if flower.classification == flowerType])
    
    def guessed_correctly(self):
        return self.classification == self.guess    

    def __str__(self):
        return "sepalLength:{0} - sepalWidth:{1} - petalLength:{2} - petalWidth:{3} - classification:{4} - guess:{5}".format(
            self.sepalLength, self.sepalWidth, self.petalLength, self.petalWidth, self.classification, self.guess or "?")
        
    def __repr__(self):
        return "sepalLength:{0} - sepalWidth:{1} - petalLength:{2} - petalWidth:{3} - classification:{4} - guess:{5}".format(
            self.sepalLength, self.sepalWidth, self.petalLength, self.petalWidth, self.classification, self.guess or "?")



class KNN:
    def __init__(self, data, k=1, controlled=True):
        self.allFlowers = copy.deepcopy(data)
        self.k = k
        self.trainingFlowers = self.controlled_shuffle() if controlled else self.shuffle()
        self.testFlowers = self.flowers_not_in_training()
        
    def shuffle(self):
        random.shuffle(self.allFlowers)
        trainingSize = (len(self.allFlowers) // 4) * 3
        training = self.allFlowers[:trainingSize]
        return training
    
    def controlled_shuffle(self):
        size = len(self.allFlowers)
        flowersPerType = int(size / len(Iris))
        trainingPerType = size // 4
        training = []
        for i in range(len(Iris)):
            start = i * flowersPerType
            end = start + flowersPerType            
            flowersInType = list(self.allFlowers[start:end])
            random.shuffle(flowersInType)
            training += flowersInType[:trainingPerType]
        return training
        
    def flowers_not_in_training(self):
        testFlowers = [flower for flower in self.allFlowers if flower not in self.trainingFlowers]
        random.shuffle(testFlowers)
        return testFlowers
        
    def run(self):
        correct_guesses = 0
        for flower in self.testFlowers:
            closeFlowers = self.get_k_flowers_close_to(flower)
            flower.make_guess(closeFlowers)
            if flower.guessed_correctly():
                correct_guesses += 1
        return (correct_guesses / len(self.testFlowers)) * 100
    
    def get_k_flowers_close_to(self, flower):
        for otherFlower in self.trainingFlowers:
            otherFlower.set_distance_to(flower)
        return sorted(self.trainingFlowers, key = lambda x: x.distance_to_current_flower)[:self.k]



def read_flower_data(filename):
    allFlowers = []
    with open(filename, 'r') as flower_data:
        for line in flower_data:
            data = re.split("\n|,", line)[:5]
            if len(data) == 5:
                flower = Flower(data[0], data[1], data[2], data[3], data[4])
                allFlowers.append(flower)
    return allFlowers

def run_iterations(controlled=False, max_k=20, runs=50):
    print("\n\nControlled Runs" if controlled else "\n\nRandom Runs")
    totalAccuracy = []
    for k in range(max_k+1):
        if (k % 2 == 0):
            continue
        results = []
        for _ in range(runs):
            iteration = KNN(data, k+1, controlled)
            results.append(iteration.run())
        accuracy = round(sum(results) / len(results))
        totalAccuracy.append(accuracy)
        print("K = " + str(k) + " -> Accuracy: " + str(accuracy) + "%")
    finalResult = round(sum(totalAccuracy) / len(totalAccuracy))
    print("Final Result = " + str(finalResult) + "%")

if __name__ == '__main__':
    data = read_flower_data('iris.data')
    run_iterations(controlled=True, max_k=9, runs=1000)
    run_iterations(controlled=False, max_k=9, runs=1000)
