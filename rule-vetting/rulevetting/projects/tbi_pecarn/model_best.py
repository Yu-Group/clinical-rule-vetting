import numpy as np
import pandas as pd
import random
import imodels
import matplotlib.pyplot as plt

from sklearn.tree import _tree, DecisionTreeClassifier, plot_tree

from rulevetting.templates.model import ModelTemplate
from rulevetting.api import validation
from rulevetting.projects.tbi_pecarn.dataset import Dataset

outcome_def = 'outcome'

class Model(ModelTemplate):
    def __init__(self):
        random.seed(10)
        df_train, df_tune, df_test = Dataset().get_data()
        features = ['HemaLoc_3', 'HAStart_2', 'HAStart_92', 'SFxPalpDepress_1',
                    'VomitNbr_1', 'VomitNbr_3', 'SFxBasRhi_0', 'LocLen_92', 'SeizOccur_2',
                    'SeizOccur_3', 'SFxBasHem_1', 'NeuroDOth_1', 'SFxBasPer_0',
                    'LOCSeparate_1', 'LOCSeparate_2', 'High_impact_InjSev_3',
                    'VomitLast_92', 'NeuroDSensory_0', 'AMS_1', 'Hema_1', 'SFxPalp_1_or_2',
                    'outcome']
        self.train = df_train[features]
        self.test = df_test[features]
        self.tune = df_tune[features]
        self.outcome_def = 'outcome'
        X_train = self.train.drop(columns=self.outcome_def)
        y_train = self.train[self.outcome_def].values
        dt = DecisionTreeClassifier(max_depth=8, class_weight={0: 1, 1: 1e2})
        self.dt = dt.fit(X_train, y_train)
        self.feature_names = self.train.drop(columns=outcome_def).columns
        
        
    
    def get_rules(self, class_names = None):
        tree_ = self.dt.tree_
        feature_names = self.feature_names
        
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []
        path = []
        proba = []
    
        def recurse(node, path, paths):
        
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [f"({name} <= {np.round(threshold, 3)})"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"({name} > {np.round(threshold, 3)})"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths += [path]
                
        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]
    
        rules = []
        for path in paths:
            rule = "if "
            
            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += str(p)
            rule += " then "
            if class_names is None:
                rule += "response: "+str(np.round(path[-1][0][0][0],3))
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                rule += f"class: {class_names(l)} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
                proba.append({np.round(classes[l]/np.sum(classes),2)})
            rule += f" | based on {path[-1][1]:,} samples"
            rules += [rule]
        return rules
     
    def get_lineage(self, df_features: pd.DataFrame):
        tree = self.dt
        feature_names = self.feature_names
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features  = [feature_names[i] for i in tree.tree_.feature]
    
         # get ids of child nodes
        idx = np.argwhere(left == -1)[:,0]     
    
        def recurse(left, right, child, lineage=None):          
            if lineage is None:
                lineage = [child]
            if child in left:
                parent = np.where(left == child)[0].item()
                split = 'l'
            else:
                parent = np.where(right == child)[0].item()
                split = 'r'
    
            lineage.append((parent, split, threshold[parent], features[parent]))
    
            if parent == 0:
                lineage.reverse()
                return lineage
            else:
                return recurse(left, right, parent, lineage)
    
        for child in idx:
            for node in recurse(left, right, child):
                print(node)    
    
    
    def tree_to_function(self):
        tree_ = self.dt.tree_
        feature_names = self.feature_names
        feature_name = [feature_names[i] 
                    if i != _tree.TREE_UNDEFINED else "undefined!" 
                    for i in tree_.feature]
        print("def tree({}):".format(", ".join(feature_names)))

        def recurse(node, depth):
            indent = "    " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                print("{}if {} <= {}:".format(indent, name, threshold))
                recurse(tree_.children_left[node], depth + 1)
                print("{}else:  # if {} > {}".format(indent, name, threshold))
                recurse(tree_.children_right[node], depth + 1)
            else:
                print("{}return {}".format(indent, np.argmax(tree_.value[node])))
    
        recurse(0, 1)   
    
    def predict(self, df_features: pd.DataFrame):
        X = df_features.drop(columns="outcome")
        preds_p = self.dt.predict_proba(X)
        Preds = 
        return preds
    
    def predict_proba(self, df_features:pd.DataFrame):
        X = df_features.drop(columns = "outcome")
        pred_prob = self.dt.predict_proba(X)
        return pred_prob
    
    def plot_model(self):
        fig = plt.figure(figsize=(100, 100))
        plot_tree(self.dt, feature_names=self.feature_names, filled=True)
        plt.show()
        
        
    def print_model(self):
        feature_names = self.feature_names
        rules = self.get_rules()
        for r in rules:
            print(r)


if __name__ == '__main__':
    from rulevetting.projects.iai_pecarn.dataset import Dataset

    df_train, df_tune, df_test = Dataset().get_data(load_csvs=True)
    df_full = pd.concat((df_train, df_tune, df_test))
    model = Model()
    preds = model.predict(df_full)
    model.print_model(df_full)
