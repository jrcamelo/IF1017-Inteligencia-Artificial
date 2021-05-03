import graphviz
from IPython.display import SVG,display
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from StarsDataset import DatasetPanda

def make_decision_tree(data, features, classes):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(data.train, data.train_classes)
    decision_tree.score(data.test, data.test_classes)
        
    train_prediction = decision_tree.predict(data.train)
    test_prediction = decision_tree.predict(data.test)
    
    print("\nResults for Training")
    print(metrics.classification_report(data.train_classes, train_prediction))
    
    print("\nResults for Test")
    
    print("\nFeature importance")
    for feature, importance in zip(features, decision_tree.feature_importances_):
        print("{}:{}".format(feature, importance))    
        
    return decision_tree, metrics.accuracy_score(data.test_classes, test_prediction)
    
def plot_tree(decision_tree, features, classes):
    dot_data = export_graphviz( 
         decision_tree, 
         out_file=None,
         feature_names=features,
         class_names=classes,  
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
    
def run_dt(target=None, plot=False):
    if target:
        data = DatasetPanda(target)
    else:
        data = DatasetPanda()
    features = data.LABELS
    classes = data.target_classes
    tree, acc = make_decision_tree(data, features, classes)
    if plot:
        plot_tree(tree, features, classes)
    return acc

def test_type():
    n = 500
    total_accuracy = 0
    for i in range(0, n):
        total_accuracy += run_dt()
    print("Total accuracy for Type = ", total_accuracy / n)
    
def test_color():
    n = 500
    total_accuracy = 0
    for i in range(0, n):
        total_accuracy += run_dt("Color")
    print("Total accuracy for Color = ", total_accuracy / n)
    

test_type()
# test_color()
