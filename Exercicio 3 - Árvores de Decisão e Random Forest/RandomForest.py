import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def execute(tree_amount=500, visualize=False, drop_columns=None):
    iris = load_iris()
    data = iris.data
    if drop_columns:
        data=pd.DataFrame({            
            'sepal length':iris.data[:,0],
            'sepal width':iris.data[:,1],
            'petal length':iris.data[:,2],
            'petal width':iris.data[:,3],
        })
        data = data.drop(columns=drop_columns)
    
    iris.names = [iris.target_names[t] for t in iris.target]
    train_inputs, test_inputs, train_classes, test_classes = train_test_split(data, iris.names, train_size=0.7)
        
    random_forest = RandomForestClassifier(
        bootstrap=True, 
        class_weight=None, 
        criterion='gini',
        max_depth=None, 
        max_features='auto', 
        max_leaf_nodes=None,
        min_impurity_decrease=0.0, 
        min_impurity_split=None,
        min_samples_leaf=1, 
        min_samples_split=2,
        min_weight_fraction_leaf=0.0, 
        n_estimators=tree_amount, 
        n_jobs=1,
        oob_score=False, 
        random_state=None, 
        verbose=0,
        warm_start=False)
    
    random_forest.fit(train_inputs, train_classes)
    random_forest.score(test_inputs, test_classes)
    
    train_prediction = random_forest.predict(train_inputs)
    test_prediction = random_forest.predict(test_inputs)
    
    if visualize:
        print("\nResults for Training")
        print(metrics.classification_report(train_classes, train_prediction))
        
        print("\nResults for Test")
        print(metrics.classification_report(test_classes, test_prediction))
        
        feature_imp = pd.Series(random_forest.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        plt.legend()
        plt.show()
    return metrics.accuracy_score(test_classes, test_prediction)

TREE_AMOUNT = 500
DROPS = [
    [],
    ["sepal length"],
    ["sepal width"],
    ["petal length"],
    ["petal width"],
    ["sepal length", "sepal width"],
    ["sepal length", "petal width"],
    ["sepal width", "petal length"],
    ["petal width", "petal length"],
    ["sepal width", "petal width"],
    ["sepal length", "petal length"],
    ["sepal width", "petal length", "petal width"],
    ["sepal length", "petal length", "petal width"],
    ["sepal length", "sepal width", "petal width"],
    ["sepal length", "sepal width", "petal length"]
]

execute(visualize=True)

executions = 100
for i in range(len(DROPS)):
    scores = 0
    for _ in range(executions):
        scores += execute(tree_amount=TREE_AMOUNT, drop_columns=DROPS[i])
    score = scores / executions
    print("Average accuracy for " + str(TREE_AMOUNT) + " trees")
    print("Removing " + str(DROPS[i]))
    print(score)


# Average accuracy for x tree amount
# 100: 0.953%
# 200: 0.951%
# 300: 0.950%
# 400: 0.953%
# 500: 0.954%
# 600: 0.954%
# 700: 0.952%
# 800: 0.953%
# 900: 0.944%
# 1000: 0.944%

# Average accuracy with combination
# SL SW PL PW: 0.952%
# SW PL PW: 0.952%
# SL PL PW: 0.944%
# SL SW PW: 0.933%
# SL SW PL: 0.933%
# PL PW: 0.958%
# SW PL: 0.933%
# SL PW: 0.927%
# SL SW: 0.710%
# SW PW: 0.930%
# SL PL: 0.933%
# SL: 0.680%
# SW: 0.501%
# PL: 0.930%
# PW: 0.952%