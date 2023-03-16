#!/usr/bin/env python
# coding: utf-8

# # Importing all necessary libraries

# In[37]:


import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import accuracy_score
import random
import math


# # Defining function to generate data

# In[38]:


def genData(x1,x2,s,label,n):
    import numpy as np
    import pandas as pd
    np.random.seed() 
    X1 = []         #list to store values of x1 attribute
    X2 = []         #list to store values of x2 attribute
    Y = [label]*n   #list to store the label for 'n' times
    
    for i in range(n):     #generating n random numbers around x1 & x2 with noise s
        X1.append(x1 + s*np.random.normal(-1,1)) 
        X2.append(x2 + s*np.random.normal(-1,1))
        
    dt = {'X1':X1, 'X2':X2, 'Y':Y}
    Data = pd.DataFrame(data=dt)
    return Data


# # Defining Node class

# In[39]:


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


# # Defining class and functions for decision tree

# In[40]:


class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2,method='gini',leafs={},l=0,centroid_1=0,centroid_neg1=0):
        
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
            possible_thresholds = np.arange(min_x,max_x,0.02)
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
        #splitting the data for left and right node
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
        return max(Y, key=Y.count) #label with maximum count assigned as value for leaf node
    
    def fit(self, X, Y):
        c1 = c2 = n1 = n2 = 0
        for i in range(len(X)):
            if(Y[i]==1):                #if the class is 1
                c1 = c1 + X[i]
                n1 = n1 + 1
            else:                       #if the class is -1
                c2 = c2 + X[i]
                n2 = n2 + 1
                
        self.centroid_1 = c1/n1     #centroid for class 1
        self.centroid_neg1 = c2/n2    #centroid for class -1
        
        
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
        #printing the tree
        if not tree:
            tree = self.root

        if tree.label is not None:
            print(tree.label)

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
            
        return c/len(Y) #number of a class label divided by total labels is the vote of that class


# In[41]:


def get_conformity_by_vote(data,sample,model):
    np.seterr(divide='ignore')
    conformity_scores = []
    p_vals = []
    
    for x in data:
        y,impurity,node = model.predict_for_sample(x,model.root)
        if(node.label==y):
            cs = node.vote
        else:
            cs = 1 - node.vote
        conformity_scores.append(cs)   #computing conformity scores of training data
        
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

        cs_1 = vote_1                   #score for current sample if class 1 is assigned
        cs_neg1 = vote_neg1             #score for current sample if class -1 is assigned
    
        ranks.append(cs_1)
        ranks = sorted(ranks,reverse=True)
        rank_1 = len(ranks) - ranks.index(cs_1) + 1     #if score is repeated in list, taking the highest index
        p_val_1 = (rank_1)/(len(ranks)+1)      #computing p-value if class is 1
        p_val.append(p_val_1)
        ranks.remove(cs_1)

        ranks.append(cs_neg1)
        ranks = sorted(ranks, reverse=True)
        rank_neg1 = len(ranks) - ranks.index(cs_neg1) + 1 #if score is repeated in list, taking the highest index
        p_val_neg1 = (rank_neg1)/(len(ranks)+1)           #computing p-value if class is -1
        p_val.append(p_val_neg1)
        ranks.remove(cs_neg1)
        
        p_vals.append(p_val)
    
    return p_vals


# In[42]:


def get_conformity_by_vote_randomized(data,sample,model):
    np.seterr(divide='ignore')
    conformity_scores = []
    p_vals = []
    
    for x in data:
        y,impurity,node = model.predict_for_sample(x,model.root)
        if(node.label==y):
            cs = node.vote
        else:
            cs = 1 - node.vote
        conformity_scores.append(cs)                #computing conformity scores of training data
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

        cs_1 = vote_1                              #score for current sample if class 1 is assigned
        cs_neg1 = vote_neg1                        #score for current sample if class -1 is assigned
        
        ranks.append(cs_1)
        ranks = sorted(ranks)
        lb = ranks.index(cs_1)
        rr = list(reversed(ranks))
        ub = len(ranks) - rr.index(cs_1) + 1                 #if score is repeated in list, taking the highest index
        p_val_1 = random.randint(lb,ub) / (len(ranks)+1)     #computing randomised p-value if class is 1
        p_val.append(p_val_1)
        ranks.remove(cs_1)
        
        ranks.append(cs_neg1)
        ranks = sorted(ranks)
        lb = ranks.index(cs_neg1)
        rr = list(reversed(ranks))
        ub = len(ranks) - rr.index(cs_neg1) + 1                  #if score is repeated in list, taking the highest index
        p_val_neg1 = random.randint(lb,ub) / (len(ranks)+1)      #computing randomised p-value if class is -1
        p_val.append(p_val_neg1)
        ranks.remove(cs_neg1)
        
        p_vals.append(p_val)
  
    return p_vals


# In[43]:


def get_conformity_by_distance(X,Y,sample,model):
    np.seterr(divide='ignore')
    conformity_scores = []
    p_vals = []

    centroid_1 = model.centroid_1
    centroid_neg1 = model.centroid_neg1
    
    for i in range(len(X)):
        d1 = math.dist(centroid_1,X[i])
        d2 = math.dist(centroid_neg1,X[i])
        
        if(Y[i]==1):
            cs = d2/d1
        else:
            cs = d1/d2
            
        conformity_scores.append(cs)                #computing conformity scores of training data
    ranks = sorted(conformity_scores)
    
    for s in sample:
        p_val = []
        d1 = math.dist(centroid_1,s)                #distance of the current sample from centroid of class 1
        d2 = math.dist(centroid_neg1,s)             #distance of the current sample from centroid of class -1
        cs_1 = d2/d1                                #score for current sample if class 1 is assigned
        cs_neg1 = d1/d2                             #score for current sample if class -1 is assigned
    
        ranks.append(cs_1)
        ranks = sorted(ranks,reverse=True)
        rank_1 = len(ranks) - ranks.index(cs_1) + 1         #if score is repeated in list, taking the highest index
        p_val_1 = (rank_1)/(len(ranks)+1)                   #computing p-value if class is 1
        p_val.append(p_val_1) 
        ranks.remove(cs_1)

        ranks.append(cs_neg1)
        ranks = sorted(ranks, reverse=True) 
        rank_neg1 = len(ranks) - ranks.index(cs_neg1) + 1   #if score is repeated in list, taking the highest index     
        p_val_neg1 = (rank_neg1)/(len(ranks)+1)             #computing p-value if class is -1
        p_val.append(p_val_neg1) 
        ranks.remove(cs_neg1)
        
        p_vals.append(p_val)

    return p_vals


# In[44]:


def get_conformity_by_distance_randomized(X,Y,sample,model):
    np.seterr(divide='ignore')
    conformity_scores = []
    p_vals = []
    centroid_1 = model.centroid_1
    centroid_neg1 = model.centroid_neg1
    
    for i in range(len(X)):
        d1 = math.dist(centroid_1,X[i])
        d2 = math.dist(centroid_neg1,X[i])
        
        if(Y[i]==1):
            cs = d2/d1
        else:
            cs = d1/d2
            
        conformity_scores.append(cs)                      #computing conformity scores of training data
    ranks = sorted(conformity_scores)
    
    for s in sample:
        p_val = []
        d1 = math.dist(centroid_1,s)                      #distance of the current sample from centroid of class 1
        d2 = math.dist(centroid_neg1,s)                   #distance of the current sample from centroid of class -1
        cs_1 = d2/d1                                      #score for current sample if class 1 is assigned
        cs_neg1 = d1/d2                                   #score for current sample if class -1 is assigned  
        
        ranks.append(cs_1)
        ranks = sorted(ranks)
        lb = ranks.index(cs_1)
        rr = list(reversed(ranks))
        ub = len(ranks) - rr.index(cs_1) + 1
        p_val_1 = random.randint(lb,ub) / (len(ranks)+1)  #if score is repeated in list, taking the highest index
        p_val.append(p_val_1)                             #computing p-value if class is 1
        ranks.remove(cs_1)
        
        ranks.append(cs_neg1)
        ranks = sorted(ranks)
        lb = ranks.index(cs_neg1)
        rr = list(reversed(ranks))
        ub = len(ranks) - rr.index(cs_neg1) + 1
        p_val_neg1 = random.randint(lb,ub) / (len(ranks)+1)   #if score is repeated in list, taking the highest index    
        p_val.append(p_val_neg1)                              #computing p-value if class is -1
        ranks.remove(cs_neg1) 
        
        p_vals.append(p_val)
    #print(conformity_scores)
    return p_vals


# In[45]:


#Standard Conformal Prediction
def std_CP(X_train,X_test,Y_test,model,CM):
    Y= []
    acc_p = []
    fal_p = []

    if(CM==1):  #CM is 1 if we want to use vote as a measure
        p_vals = get_conformity_by_vote(X_train,X_test,model)
    elif(CM==2):  #CM is 2 if we want to use distance to the centroid as a measure
        p_vals = get_conformity_by_distance(X_train,Y_train,X_test,model)
    else:
        print('Wrong choise for conformity measure!')

    for i in range(len(p_vals)):
        y = 1 if p_vals[i][0]>p_vals[i][1] else -1        #making prediction for current sample based on largest p-value
        Y.append(y)
        t = p_vals[i][0] if p_vals[i][0]>p_vals[i][1] else p_vals[i][1]
        if(y==Y_test[i]):                                 #checking true label y and storing current p-value accordingly
            acc_p.append(t)
        else:
            fal_p.append(t)

    return fal_p,Y,acc_p


# In[46]:


#Randomized CP
def randomized_CP(X_train,X_test,Y_test,model,CM):
    Y= []
    acc_p = []
    fal_p = []
    
    if(CM==1):     #CM is 1 if we want to use vote as a measure
        p_vals = get_conformity_by_vote_randomized(X_train,X_test,model)
    elif(CM==2):   #CM is 2 if we want to use distance to the centroid as a measure
        p_vals = get_conformity_by_distance_randomized(X_train,Y_train,X_test,model)
    else:
        print('Wrong choise for conformity measure!')

    for i in range(len(p_vals)):
        y = 1 if p_vals[i][0]>p_vals[i][1] else -1        #making prediction for current sample based on largest p-value
        Y.append(y)
        t = p_vals[i][0] if p_vals[i][0]>p_vals[i][1] else p_vals[i][1]
        if(y==Y_test[i]):                                 #checking true label y and storing current p-value accordingly
            acc_p.append(t)
        else:
            fal_p.append(t)

    return fal_p,Y,acc_p


# In[47]:


#Increasing Noise
def IncreasingNoise():
    #Below lists will be used for CP with 1st Conformity measure
    val_std = []
    val_ran = []
    acc_std = []
    acc_ran = []
    eff_std = []
    eff_ran = []
    
    #Below lists will be used for CP with 2nd Conformity measure
    val_std2 = []
    val_ran2 = []
    acc_std2 = []
    acc_ran2 = []
    eff_std2 = []
    eff_ran2 = []
    
    Train_class1 = genData(2,3,1,1,120)
    Train_class2 = genData(0,0,2,-1,120)
    train_data = pd.concat([Train_class1, Train_class2])
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    X_train = train_data.iloc[:, :-1].values
    Y_train = train_data.iloc[:, -1].values.reshape(-1,1)

    #Training Decision Tree
    model = DecisionTreeClassifier(method='entropy',max_depth=2)
    model.fit(X_train,Y_train)

    for s in range(0,10):
        #Generating test data
        Test_class1 = genData(2,3,1+s,1,80)
        Test_class2 = genData(0,0,2+s,-1,80)
        test_data = pd.concat([Test_class1, Test_class2])
        test_data = test_data.sample(frac=1).reset_index(drop=True)
        X_test = test_data.iloc[:, :-1].values
        Y_test = test_data.iloc[:, -1].values.reshape(-1,1)
        
        #Using standard CP and randomized CP with 1st conformity measure
        fal_p,Y_std,acc_p = std_CP(X_train,X_test,Y_test,model,1)
        val_std.append(sum(acc_p)/len(acc_p))    #Validity
        eff_std.append(sum(fal_p)/len(fal_p))    #Efficiency
        acc_std.append(accuracy_score(Y_test,Y_std))
    
        fal_p,Y_ran,acc_p = randomized_CP(X_train,X_test,Y_test,model,1)
        val_ran.append(sum(acc_p)/len(acc_p))    #Validity
        eff_ran.append(sum(fal_p)/len(fal_p))    #Efficiency
        acc_ran.append(accuracy_score(Y_test,Y_ran))
        
        
        #Using standard CP and randomized CP with 2nd conformity measure
        fal_p,Y_std,acc_p = std_CP(X_train,X_test,Y_test,model,2)
        val_std2.append(sum(acc_p)/len(acc_p))    #Validity
        eff_std2.append(sum(fal_p)/len(fal_p))    #Efficiency
        acc_std2.append(accuracy_score(Y_test,Y_std))
    
        fal_p,Y_ran,acc_p = randomized_CP(X_train,X_test,Y_test,model,2)
        val_ran2.append(sum(acc_p)/len(acc_p))    #Validity
        eff_ran2.append(sum(fal_p)/len(fal_p))    #Efficiency
        acc_ran2.append(accuracy_score(Y_test,Y_ran))
        
    result_std1 = pd.DataFrame({'Validity': val_std, 'Efficiency' : eff_std, 'Accuracy' : acc_std})
    result_ran1 = pd.DataFrame({'Validity': val_ran, 'Efficiency' : eff_ran, 'Accuracy' : acc_ran})
    result_std2 = pd.DataFrame({'Validity': val_std2, 'Efficiency' : eff_std2, 'Accuracy' : acc_std2})
    result_ran2 = pd.DataFrame({'Validity': val_ran2, 'Efficiency' : eff_ran2, 'Accuracy' : acc_ran2})
    
    return result_std1, result_ran1, result_std2, result_ran2


# In[48]:


#Decreasing Noise
def DecreasingNoise():
    #Below lists will be used for CP with 1st Conformity measure
    val_std = []
    val_ran = []
    acc_std = []
    acc_ran = []
    eff_std = []
    eff_ran = []
    
    #Below lists will be used for CP with 2nd Conformity measure
    val_std2 = []
    val_ran2 = []
    acc_std2 = []
    acc_ran2 = []
    eff_std2 = []
    eff_ran2 = []
    
    Train_class1 = genData(2,3,9,1,120)
    Train_class2 = genData(0,0,10,-1,120)
    train_data = pd.concat([Train_class1, Train_class2])
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    X_train = train_data.iloc[:, :-1].values
    Y_train = train_data.iloc[:, -1].values.reshape(-1,1)

    #Training Decision Tree
    model = DecisionTreeClassifier(method='entropy',max_depth=2)
    model.fit(X_train,Y_train)
    

    for s in range(10,0,-1):
        Test_class1 = genData(2,3,s-1,1,80)
        Test_class2 = genData(0,0,s,-1,80)
        test_data = pd.concat([Test_class1, Test_class2])
        test_data = test_data.sample(frac=1).reset_index(drop=True)
        X_test = test_data.iloc[:, :-1].values
        Y_test = test_data.iloc[:, -1].values.reshape(-1,1)
    
        #Using standard CP and randomized CP with 1st conformity measure
        fal_p,Y_std,acc_p = std_CP(X_train,X_test,Y_test,model,1)
        val_std.append(sum(acc_p)/len(acc_p))    #Validity
        eff_std.append(sum(fal_p)/len(fal_p))    #Efficiency
        acc_std.append(accuracy_score(Y_test,Y_std))
    
        fal_p,Y_ran,acc_p = randomized_CP(X_train,X_test,Y_test,model,1)
        val_ran.append(sum(acc_p)/len(acc_p))    #Validity
        eff_ran.append(sum(fal_p)/len(fal_p))    #Efficiency
        acc_ran.append(accuracy_score(Y_test,Y_ran))
        
        
        #Using standard CP and randomized CP with 2nd conformity measure
        fal_p,Y_std,acc_p = std_CP(X_train,X_test,Y_test,model,2)
        val_std2.append(sum(acc_p)/len(acc_p))    #Validity
        eff_std2.append(sum(fal_p)/len(fal_p))    #Efficiency
        acc_std2.append(accuracy_score(Y_test,Y_std))
    
        fal_p,Y_ran,acc_p = randomized_CP(X_train,X_test,Y_test,model,2)
        val_ran2.append(sum(acc_p)/len(acc_p))    #Validity
        eff_ran2.append(sum(fal_p)/len(fal_p))    #Efficiency
        acc_ran2.append(accuracy_score(Y_test,Y_ran))
        
    result_std1 = pd.DataFrame({'Validity': val_std, 'Efficiency' : eff_std, 'Accuracy' : acc_std})
    result_ran1 = pd.DataFrame({'Validity': val_ran, 'Efficiency' : eff_ran, 'Accuracy' : acc_ran})
    result_std2 = pd.DataFrame({'Validity': val_std2, 'Efficiency' : eff_std2, 'Accuracy' : acc_std2})
    result_ran2 = pd.DataFrame({'Validity': val_ran2, 'Efficiency' : eff_ran2, 'Accuracy' : acc_ran2})
    
    return result_std1, result_ran1, result_std2, result_ran2


# In[49]:


#shifting models
def ShiftingModel():
    #Below lists will be used for CP with 1st Conformity measure
    val_std = []
    val_ran = []
    acc_std = []
    acc_ran = []
    eff_std = []
    eff_ran = []
    
    #Below lists will be used for CP with 2nd Conformity measure
    val_std2 = []
    val_ran2 = []
    acc_std2 = []
    acc_ran2 = []
    eff_std2 = []
    eff_ran2 = []
    
    Train_class1 = genData(2,3,1,1,120)
    Train_class2 = genData(0,0,2,-1,120)
    train_data = pd.concat([Train_class1, Train_class2])
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    X_train = train_data.iloc[:, :-1].values
    Y_train = train_data.iloc[:, -1].values.reshape(-1,1)

    model = DecisionTreeClassifier(method='entropy',max_depth=2)
    model.fit(X_train,Y_train)

    for n in range(0,10):
        x_pos = 2+ n
        x_neg = 0+ n
        Test_class1 = genData(x_pos,x_pos+1,1,1,80)
        Test_class2 = genData(x_neg,x_neg,2,-1,80)
        test_data = pd.concat([Test_class1, Test_class2])
        test_data = test_data.sample(frac=1).reset_index(drop=True)
        X_test = test_data.iloc[:, :-1].values
        Y_test = test_data.iloc[:, -1].values.reshape(-1,1)
        
        #Using standard CP and randomized CP with 1st conformity measure
        fal_p,Y_std,acc_p = std_CP(X_train,X_test,Y_test,model,1)
        val_std.append(sum(acc_p)/len(acc_p))    #Validity
        eff_std.append(sum(fal_p)/len(fal_p))    #Efficiency
        acc_std.append(accuracy_score(Y_test,Y_std))
    
        fal_p,Y_ran,acc_p = randomized_CP(X_train,X_test,Y_test,model,1)
        val_ran.append(sum(acc_p)/len(acc_p))    #Validity
        eff_ran.append(sum(fal_p)/len(fal_p))    #Efficiency
        acc_ran.append(accuracy_score(Y_test,Y_ran))
        
        
        #Using standard CP and randomized CP with 2nd conformity measure
        fal_p,Y_std,acc_p = std_CP(X_train,X_test,Y_test,model,2)
        val_std2.append(sum(acc_p)/len(acc_p))    #Validity
        eff_std2.append(sum(fal_p)/len(fal_p))    #Efficiency
        acc_std2.append(accuracy_score(Y_test,Y_std))
    
        fal_p,Y_ran,acc_p = randomized_CP(X_train,X_test,Y_test,model,2)
        val_ran2.append(sum(acc_p)/len(acc_p))    #Validity
        eff_ran2.append(sum(fal_p)/len(fal_p))    #Efficiency
        acc_ran2.append(accuracy_score(Y_test,Y_ran))
        
    result_std1 = pd.DataFrame({'Validity': val_std, 'Efficiency' : eff_std, 'Accuracy' : acc_std})
    result_ran1 = pd.DataFrame({'Validity': val_ran, 'Efficiency' : eff_ran, 'Accuracy' : acc_ran})
    result_std2 = pd.DataFrame({'Validity': val_std2, 'Efficiency' : eff_std2, 'Accuracy' : acc_std2})
    result_ran2 = pd.DataFrame({'Validity': val_ran2, 'Efficiency' : eff_ran2, 'Accuracy' : acc_ran2})
    
    return result_std1, result_ran1, result_std2, result_ran2


# # Building decision tree without conformal prediction

# In[50]:



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
print('Accuracy: '+str(accuracy_score(Y_test,Y)))


# # Results of all four models when the transfer case is increasing noise in test data

# In[54]:


result_std1, result_ran1, result_std2, result_ran2 = IncreasingNoise()
print('Standard Conformal Prediction with measure as vote')
print(result_std1)
print('--------------------------- \n Randomised Conformal Prediction with measure as vote')
print(result_ran1)
print('--------------------------- \n Standard Conformal Prediction with measure as distance to the centroid')
print(result_std2)
print('--------------------------- \n Randomised Conformal Prediction with measure as distance to the centroid')
print(result_ran2)


# # Results of all four models when the transfer case is decreasing noise in test data

# In[52]:


result_std1, result_ran1, result_std2, result_ran2  = DecreasingNoise()
print('Standard Conformal Prediction with measure as vote')
print(result_std1)
print('--------------------------- \n Randomised Conformal Prediction with measure as vote')
print(result_ran1)
print('--------------------------- \n Standard Conformal Prediction with measure as distance to the centroid')
print(result_std2)
print('--------------------------- \n Randomised Conformal Prediction with measure as distance to the centroid')
print(result_ran2)


# # Results of all four models when the transfer case is increasing the shift test data

# In[55]:


result_std1, result_ran1, result_std2, result_ran2 = ShiftingModel()
print('Standard Conformal Prediction with measure as vote')
print(result_std1)
print('--------------------------- \n Randomised Conformal Prediction with measure as vote')
print(result_ran1)
print('--------------------------- \n Standard Conformal Prediction with measure as distance to the centroid')
print(result_std2)
print('--------------------------- \n Randomised Conformal Prediction with measure as distance to the centroid')
print(result_ran2)

