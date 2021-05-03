import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from StarsDataset import DatasetPanda, CLASSIFICATION

from dtreeviz.trees import dtreeviz

def execute(tree_amount=500, visualize=False, drop_labels=None):
    data = DatasetPanda()
    data.drop_labels(drop_labels)
    
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
    
    random_forest.fit(data.train, data.train_classes)
    random_forest.score(data.test, data.test_classes)
    
    train_prediction = random_forest.predict(data.train)
    test_prediction = random_forest.predict(data.test)
    
    if visualize:
        print("\nResults for Training")
        print(metrics.classification_report(data.train_classes, train_prediction))
        
        print("\nResults for Test")
        print(metrics.classification_report(data.test_classes, test_prediction))
        
        feature_imp = pd.Series(random_forest.feature_importances_, index=data.LABELS)
        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        plt.legend()
        plt.show()
        
        viz = dtreeviz(random_forest.estimators_[0], data.data, data.target,
                target_name="target",
                feature_names=data.LABELS,
                class_names=CLASSIFICATION)
        viz.save("decision_tree.svg")
    return metrics.accuracy_score(data.test_classes, test_prediction)

TREE_AMOUNT = 50
DROPS = [
    [],
]

execute(visualize=True)

executions = 10
for i in range(len(DROPS)):
    scores = 0
    for _ in range(executions):
        scores += execute(tree_amount=TREE_AMOUNT, drop_labels=DROPS[i])
    score = scores / executions
    print("Average accuracy for " + str(TREE_AMOUNT) + " trees")
    print("Removing " + str(DROPS[i]))
    print(score)


# NEED TO EXPERIMENT
# NEED TO CREATE VISUALIZATIONS
# NEED TO DROP FEATURES
