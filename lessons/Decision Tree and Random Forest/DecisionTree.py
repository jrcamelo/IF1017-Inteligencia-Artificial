import graphviz
from IPython.display import SVG,display
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

iris = load_iris()
iris.names = [iris.target_names[t] for t in iris.target]
train_inputs, test_inputs, train_classes, test_classes = train_test_split(iris.data, iris.names, train_size=0.7)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_inputs, train_classes)
decision_tree.score(test_inputs, test_classes)
    
train_prediction = decision_tree.predict(train_inputs)
test_prediction = decision_tree.predict(test_inputs)

print("\nResults for Training")
print(metrics.classification_report(train_classes, train_prediction))

print("\nResults for Test")
print(metrics.classification_report(test_classes, test_prediction))

print("\nFeature importance")
for feature, importance in zip(iris.feature_names, decision_tree.feature_importances_):
    print("{}:{}".format(feature, importance))

dot_data = export_graphviz( 
         decision_tree, 
         out_file=None,
         feature_names=iris.feature_names,
         class_names=iris.target_names,  
         filled=True, 
         rounded=True,
         proportion=True,
         node_ids=True,
         rotate=False,
         label='all',
         special_characters=True
        )
graph = graphviz.Source(dot_data)  
display(SVG(graph.pipe(format='svg')))