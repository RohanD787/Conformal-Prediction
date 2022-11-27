#!/usr/bin/env python
# coding: utf-8

# In[1]:


def genData(x1,x2,s,label,n):
    import numpy as np
    import pandas as pd
    np.random.seed()
    X1 = []
    X2 = []
    #X = []
    Y = [label]*n
    for i in range(n):
        X1.append(x1 + s*np.random.normal(0,1))
        X2.append(x2 + s*np.random.normal(0,1))

        #f1 = x1 + s*np.random.normal(0,1)
       # f2 = x2 + s*np.random.normal(0,1)
        #X.append([f1,f2])
        
    dt = {'X1':X1, 'X2':X2, 'Y':Y}
    Data = pd.DataFrame(data=dt)
    return Data


# In[2]:


class Node():
    
    def __init__(self, X_index=None, threshold=None, left=None, right=None, info_gain=None, label=None, impurity=0,vote=0):
        
            self.X_index = X_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.info_gain = info_gain
            self.leaf = False
            

            self.label = label
            self.leaf = True
            self.impurity = impurity
            self.vote = vote


# In[3]:


class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2,method='gini',leafs={},l=0):
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
        self.method = method
        self.leafs = leafs
        self.l=l
        
    def build_tree(self, dataset, curr_depth=0):

        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        
        #storing impurity and vote of leaf
        v = self.get_vote(leaf_value,Y)
        if(self.method=='gini'):
            imp = self.gini_index(Y)
        else:
            imp = self.entropy(Y)
            
        # return leaf node
        return Node(label=leaf_value,impurity=imp,vote=v)
    
    def get_best_split(self, dataset, n, Xn):
        
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for Xi in range(Xn):
            x = dataset[:, Xi]
            
            max_x = np.max(x)
            min_x = np.min(x)
            possible_thresholds = np.arange(min_x,max_x,0.2)
            #possible_thresholds = np.unique(x)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, Xi, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.info_gain(y, left_y, right_y, self.method)
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = Xi
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
        
    def info_gain(self, parent, l_child, r_child, method='gini'):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        
        if method=='gini':    #computing info gain with gini index
            info_gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
            
        elif method=='entropy':                #computing info gain with entropy
            info_gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
            
        return info_gain
    
    def entropy(self, Y):
        labels = np.unique(Y)
        entropy = 0
        for l in labels:
            p_cls = len(Y[Y == l]) / len(Y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, Y):
        labels = np.unique(Y)
        gini = 0
        for l in labels:
            p_cls = len(Y[Y == l]) / len(Y)
            gini += p_cls**2
        return 1 - gini
    
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        
    def predict_for_data(self,X):
        Y = []
        for x in X:
            y = self.predict_for_x(x,self.root)     #passing x and root of tree, getting prediction y for x
            Y.append(y)
        return Y
    
    def predict_for_x(self,X,node):
        #similar to 'predict_for_sample', but only used on test data without using CP
        
        if(node.label!=None):       #if node is leaf node, returning it's class
            return node.label
        
        x = X[node.X_index]
        
        if(x<=node.threshold):
            return self.predict_for_x(X,node.left)    #if x<=threshold, going to left node
        else:
            return self.predict_for_x(X,node.right)   #if x>threshold, going to right node
    
    def predict_for_sample(self,X,node):
        #similar to predict_for_x, but used while applying CP
        
        if(node.label!=None):       #if node is leaf node, returning it's class
            return node.label, node.impurity, node
        
        x = X[node.X_index]
        
        if(x<=node.threshold):
            return self.predict_for_sample(X,node.left)    #if x<=threshold, going to left node
        else:
            return self.predict_for_sample(X,node.right)   #if x>threshold, going to right node
        
    def print_tree(self, tree=None, indent=" "):
        
        if not tree:
            tree = self.root

        if tree.label is not None:
            print(tree.label)
            print(f'node_num:{tree.num}')
            print(f'Impurity: {tree.impurity}')
            print(f'Vote: {tree.vote}')

        else:
            print("X_"+str(tree.X_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    

    def get_vote(self,label,Y):
        c = 0
        for y in Y:
            if y==label:
                c+=1
            
        return c/len(Y)


# In[4]:


def get_conformity_by_vote(data,sample,model):
    np.seterr(divide='ignore')
    conformity_scores = []
    p_vals = []
    
    for x in data:
        label,impurity,node = model.predict_for_sample(x,model.root)
        cs = node.vote
        conformity_scores.append(cs)
    ranks = sorted(conformity_scores)
    #print(ranks)
    
    for i in range(len(sample)):
        label,impurity,node = model.predict_for_sample(sample[i],model.root)
        p_val = []
        if(label==1):
            vote_1 = node.vote
            vote_neg1 = 1 - node.vote
    
        else:
            vote_1 = 1 - node.vote
            vote_neg1 = node.vote

        cs_1 = vote_1
        cs_neg1 = vote_neg1
    
        ranks.append(cs_1)
        ranks = sorted(ranks,reverse=True)
        rank_1 = len(ranks) - ranks.index(cs_1) + 1
        p_val_1 = (rank_1)/(len(ranks)+1)
        p_val.append(p_val_1)
        ranks.remove(cs_1)

        ranks.append(cs_neg1)
        ranks = sorted(ranks, reverse=True)
        rank_neg1 = len(ranks) - ranks.index(cs_neg1) + 1
        p_val_neg1 = (rank_neg1)/(len(ranks)+1)
        p_val.append(p_val_neg1)
        ranks.remove(cs_neg1)
        
        p_vals.append(p_val)
    #print(conformity_scores)
    return p_vals


# In[5]:


def get_conformity_by_vote_randomized(data,sample,model):
    np.seterr(divide='ignore')
    conformity_scores = []
    p_vals = []
    
    for x in data:
        label,impurity,node = model.predict_for_sample(x,model.root)
        cs = node.vote
        conformity_scores.append(cs)
    ranks = sorted(conformity_scores)
    
    for i in range(len(sample)):
        label,impurity,node = model.predict_for_sample(sample[i],model.root)
        p_val = []
        if(label==1):
            vote_1 = node.vote
            vote_neg1 = 1 - node.vote
    
        else:
            vote_1 = 1 - node.vote
            vote_neg1 = node.vote

        cs_1 = vote_1
        cs_neg1 = vote_neg1
        
        ranks.append(cs_1)
        ranks = sorted(ranks)
        lb = ranks.index(cs_1)
        rr = list(reversed(ranks))
        ub = len(ranks) - rr.index(cs_1) + 1
        p_val_1 = random.randint(lb,ub) / (len(ranks)+1)
        p_val.append(p_val_1)
        ranks.remove(cs_1)
        
        ranks.append(cs_neg1)
        ranks = sorted(ranks)
        lb = ranks.index(cs_neg1)
        rr = list(reversed(ranks))
        ub = len(ranks) - rr.index(cs_neg1) + 1
        p_val_neg1 = random.randint(lb,ub) / (len(ranks)+1)      
        p_val.append(p_val_neg1)
        ranks.remove(cs_neg1)
        
        p_vals.append(p_val)
    #print(conformity_scores)
    return p_vals


# In[6]:


#Generating Training Data And Building Decision Tree

import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import accuracy_score
n=300

#Generating training data
Train_class1 = genData(2,3,1,1,120)
Train_class2 = genData(0,0,2,-1,120)
train_data = pd.concat([Train_class1, Train_class2])
train_data = train_data.sample(frac=1).reset_index(drop=True)
X_train = train_data.iloc[:, :-1].values
Y_train = train_data.iloc[:, -1].values.reshape(-1,1)

#Generating data to test CP
Test_class1 = genData(2,3,1,1,80)
Test_class2 = genData(0,0,2,-1,80)
test_data = pd.concat([Test_class1, Test_class2])
test_data = test_data.sample(frac=1).reset_index(drop=True)
X_test = test_data.iloc[:, :-1].values
Y_test = test_data.iloc[:, -1].values.reshape(-1,1)
data = X_train
sample = X_test

model = DecisionTreeClassifier(method='entropy')
model.fit(X_train,Y_train)
Y = model.predict_for_data(X_test)
print(accuracy_score(Y_test,Y))


# In[7]:


#Standard and Randomized Conformal Prediction
def std_CP(data,sample,model):
    p_vals = get_conformity_by_vote(data,sample,model)
    Y= []
    acc_p = []

    for i in range(len(p_vals)):
        y = 1 if p_vals[i][0]>p_vals[i][1] else -1
        Y.append(y)
        if(y==Y_test[i]):
            t = p_vals[i][0] if p_vals[i][0]>p_vals[i][1] else p_vals[i][1]
            acc_p.append(t)

    return p_vals,Y,acc_p


# In[8]:


#Standard and Randomized CP with shift in model
def randomized_CP(data,sample,model):
    p_vals = get_conformity_by_vote_randomized(data,sample,model)
    Y= []
    acc_p = []

    for i in range(len(p_vals)):
        y = 1 if p_vals[i][0]>p_vals[i][1] else -1
        Y.append(y)
        if(y==Y_test[i]):
            t = p_vals[i][0] if p_vals[i][0]>p_vals[i][1] else p_vals[i][1]
            acc_p.append(t)

    return p_vals,Y,acc_p


# In[9]:


#Decreasing Noise
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import accuracy_score
import random

Train_class1 = genData(2,3,1,1,120)
Train_class2 = genData(0,0,2,-1,120)
train_data = pd.concat([Train_class1, Train_class2])
train_data = train_data.sample(frac=1).reset_index(drop=True)
X_train = train_data.iloc[:, :-1].values
Y_train = train_data.iloc[:, -1].values.reshape(-1,1)

#Training Decision Tree
model_1 = DecisionTreeClassifier(method='entropy',max_depth=2)
model_1.fit(X_train,Y_train)

val_std = []
val_ran = []
acc_std = []
acc_ran = []

for s in range(10,0,-1):
    Test_class1 = genData(2,3,s-1,1,80)
    Test_class2 = genData(0,0,s,-1,80)
    test_data = pd.concat([Test_class1, Test_class2])
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    X_test = test_data.iloc[:, :-1].values
    Y_test = test_data.iloc[:, -1].values.reshape(-1,1)
    
    p_vals,Y_std,acc_p = std_CP(X_train,X_test,model_1)
    val_std.append(sum(acc_p)/len(acc_p))    #Validity
    acc_std.append(accuracy_score(Y_test,Y_std))
    
    p_vals,Y_ran,acc_p = randomized_CP(X_train,X_test,model_1)
    val_ran.append(sum(acc_p)/len(acc_p))    #Validity
    acc_ran.append(accuracy_score(Y_test,Y_ran))


# In[10]:





# In[11]:


#shifting models
Train_class1 = genData(2,3,1,1,120)
Train_class2 = genData(0,0,2,-1,120)
train_data = pd.concat([Train_class1, Train_class2])
train_data = train_data.sample(frac=1).reset_index(drop=True)
X_train = train_data.iloc[:, :-1].values
Y_train = train_data.iloc[:, -1].values.reshape(-1,1)

model_2 = DecisionTreeClassifier(method='entropy',max_depth=2)
model_2.fit(X_train,Y_train)

val_std = []
val_ran = []
acc_std = []
acc_ran = []
shifts = []

for n in range(0,10):
    x_pos = 2+n
    x_neg = 0+n
    shifts.append([(x_pos,x_pos+1),(x_neg,x_neg)])
    Test_class1 = genData(x_pos,x_pos+1,1,1,80)
    Test_class2 = genData(x_neg,x_neg,2,-1,80)
    test_data = pd.concat([Test_class1, Test_class2])
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    X_test = test_data.iloc[:, :-1].values
    Y_test = test_data.iloc[:, -1].values.reshape(-1,1)
    
    p_vals,Y_std,acc_p = std_CP(X_train,X_test,model_1)
    val_std.append(sum(acc_p)/len(acc_p))    #Validity
    acc_std.append(accuracy_score(Y_test,Y_std))
    
    p_vals,Y,acc_p = randomized_CP(X_train,X_test,model_1)
    val_ran.append(sum(acc_p)/len(acc_p))    #Validity
    acc_ran.append(accuracy_score(Y_test,Y_ran))


# In[12]:





# In[13]:


#Increasing Noise
Train_class1 = genData(2,3,1,1,120)
Train_class2 = genData(0,0,2,-1,120)
train_data = pd.concat([Train_class1, Train_class2])
train_data = train_data.sample(frac=1).reset_index(drop=True)
X_train = train_data.iloc[:, :-1].values
Y_train = train_data.iloc[:, -1].values.reshape(-1,1)

#Training Decision Tree
model_1 = DecisionTreeClassifier(method='entropy',max_depth=2)
model_1.fit(X_train,Y_train)

val_std = []
val_ran = []
acc_std = []
acc_ran = []

for s in range(0,10):
    Test_class1 = genData(2,3,1+s,1,80)
    Test_class2 = genData(0,0,2+s,-1,80)
    test_data = pd.concat([Test_class1, Test_class2])
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    X_test = test_data.iloc[:, :-1].values
    Y_test = test_data.iloc[:, -1].values.reshape(-1,1)
    
    p_vals,Y_std,acc_p = std_CP(X_train,X_test,model_1)
    val_std.append(sum(acc_p)/len(acc_p))    #Validity
    acc_std.append(accuracy_score(Y_test,Y_std))
    
    p_vals,Y_ran,acc_p = randomized_CP(X_train,X_test,model_1)
    val_ran.append(sum(acc_p)/len(acc_p))    #Validity
    acc_ran.append(accuracy_score(Y_test,Y_ran))


# In[ ]:




