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
    print(metrics.classification_report(data.test_classes, test_prediction))
    
    print("\nFeature importance")
    for feature, importance in zip(features, decision_tree.feature_importances_):
        print("{}:{}".format(feature, importance))    
    return decision_tree
    
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
    
def run_for_type():
    data = DatasetPanda()
    features = data.LABELS
    classes = data.target_classes
    tree = make_decision_tree(data, features, classes)
    plot_tree(tree, features, classes)
    
def run_for_color():
    data = DatasetPanda(target="Color")
    print(data.data)
    features = data.LABELS
    classes = data.target_classes
    tree = make_decision_tree(data, features, classes)
    plot_tree(tree, features, classes)

run_for_type()
run_for_color()

# NEED TO EXPERIMENT
# NEED TO PRUNE TREE
# NEED TO MAKE PATH IT TOOK
