import re
from matplotlib import pyplot
from matplotlib.colors import ListedColormap
import numpy

class Iris:        
    def __init__(self, label, value, marker, color):
        self.label = label
        self.value = value
        self.marker = marker
        self.color = color
        
    def __str__(self):
        return self.label
    
    def SETOSA():
        return Iris("Iris-setosa", -1, "o", "red")
    def VERSICOLOR():
        return Iris("Iris-versicolor", 1, "x", "blue")

class Flower:    
    def __init__(self, 
                 sepalLength, 
                 sepalWidth, 
                 petalLength, 
                 petalWidth, 
                 label):
        self.sepalLength = float(sepalLength)
        self.petalLength = float(petalLength)
        self.lengths = self.make_length_ndarray()
        self.label = label
        self.value = self.get_value()
        
    def get_value(self):
        if Iris.SETOSA().label == self.label:
            return Iris.SETOSA().value
        if Iris.VERSICOLOR().label == self.label:
            return Iris.VERSICOLOR().value
        return None
    
    def make_length_ndarray(self):
        shape = numpy.shape([self.sepalLength, self.petalLength])
        return numpy.ndarray(shape)
                            
    def __str__(self):
        return "sepalLength:{0} - petalLength:{1} - label:{2} - value:{3}".format(
            self.sepalLength, self.petalLength, self.label, self.value or "?")
        
    def __repr__(self):
        return "sepalLength:{0} - petalLength:{1} - label:{2} - value:{3}".format(
            self.sepalLength, self.petalLength, self.label, self.value or "?")


class FlowerList(list):    
    def __init__(self, list_of_flowers=None):
        super()
        if list_of_flowers:
            self.extend(list_of_flowers)

    def load_file(self, filename):
        with open(filename, 'r') as flower_data:
            for line in flower_data:
                self.__parse_and_append_flower(line)
        return self
    
    def __parse_and_append_flower(self, line):
        data = re.split("\n|,", line)[:5]
        if len(data) == 5:
            flower = Flower(data[0], data[1], data[2], data[3], data[4])
            if flower.value:
                self.append(flower)
            
    def values(self):
        return [flower.value for flower in self]
    
    def sepal_lengths(self):
        return [flower.sepalLength for flower in self]
    
    def petal_lengths(self):
        return [flower.petalLength for flower in self]
    
    def lengths(self):
        shape = numpy.shape([flower.lengths for flower in self])
        return numpy.ndarray(shape)
    
    def get_only_label(self, label):
        return FlowerList([flower for flower in self if flower.label == label])
    
    def make_training_and_test(self):
        flowersPerType = int(len(self) / 2)
        trainingPerType = len(self) // 3
        training = []
        for i in range(2):
            start = i * flowersPerType
            end = start + flowersPerType            
            flowersInType = list(self[start:end])
            training += flowersInType[:trainingPerType]
        test = [flower for flower in self if flower not in training]
        return FlowerList(training), FlowerList(test)

class FlowerDataset:
    def __init__(self, filename, should_train=True):
        self.allFlowers = FlowerList().load_file(filename)
        self.training, self.test = self.allFlowers.make_training_and_test()
        self.flowers = self.training if should_train else self.test
    
    def set_training(self, should_train=True):
        self.flowers = self.training if should_train else self.test
        return self

    def plot_flowers(self):
        setosa = self.flowers.get_only_label(Iris.SETOSA().label)
        pyplot.scatter(setosa.sepal_lengths(), 
                       setosa.petal_lengths(),
                       color=Iris.SETOSA().color,
                       marker=Iris.SETOSA().marker,
                       label=Iris.SETOSA().label)
        versicolor = self.flowers.get_only_label(Iris.VERSICOLOR().label)
        pyplot.scatter(versicolor.sepal_lengths(),
                       versicolor.petal_lengths(),
                       color=Iris.VERSICOLOR().color,
                       marker=Iris.VERSICOLOR().marker,
                       label=Iris.VERSICOLOR().label)
        pyplot.xlabel("Petal Length")
        pyplot.ylabel("Sepal Length")
        pyplot.legend(loc="upper left")
        pyplot.show()      
    
    # Too much to make readable
    def plot_decision_regions(self, classifier, resolution=0.02):
        X = self.flowers.lengths()
        y = self.flowers.values()
        
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(numpy.unique(y))])
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx1, xx2 = numpy.meshgrid(numpy.arange(x1_min, x1_max, resolution), numpy.arange(x2_min, x2_max, resolution))
        
        Z = classifier.predict(numpy.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        pyplot.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        
        for idx, cl in enumerate(numpy.unique(y)):
            pyplot.scatter(x=X[y == cl, 0], 
                           y=X[y==cl, 1], 
                        alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl)
        pyplot.xlabel("Petal Length")
        pyplot.ylabel("Sepal Length")
        pyplot.legend(loc="upper left")
        pyplot.show()      

class Perceptron:
    def __init__(self, data, learningRate=0.01, iterations=10):
        self.data = data
        self.flowers = data.flowers
        self.learningRate = learningRate
        self.iterations = iterations
        self.types = 2
        
    def set_test_flowers(self):
        data.set_training(False)
        self.flowers = data.flowers
    
    def fit(self):
        self.weights = [0] * (self.types + 1)
        self.totalErrors = []
        for _ in range(self.iterations):
            self.run_fitting_iteration()
    
    def run_fitting_iteration(self):
        errors = 0
        for flower in self.flowers:
            error = self.calculate_error(flower)
            self.weights[0] += error
            self.weights[1:] += error * flower.lengths
            errors += 1 if error else 0
        self.totalErrors.append(errors)
    
    def calculate_error(self, flower):
        prediction = self.predict(flower.lengths)
        return self.learningRate * (flower.value - prediction)
    
    def predict(self, inputs):
        net_is_positive = self.net_input(inputs) >= 0.0
        return numpy.where(net_is_positive, 1, -1)
    
    def net_input(self, inputs):
        return numpy.dot(inputs, self.weights[1:]) + self.weights[0]
            
        
if __name__ == '__main__':
    data = FlowerDataset('iris.data')
    ppn = Perceptron(data, 0.10, 100)
    ppn.fit()
    data.plot_decision_regions(ppn)
    ppn.set_test_flowers()
    data.plot_decision_regions(ppn)
    pyplot.show()
