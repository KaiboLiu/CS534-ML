"""
Created on Fri Nov 04 23:01:07 2016

@author: Kaibo Liu
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import pdb
import copy
import collections  # do statistics on Leaf node

import LoadData   as ld
# import OutputData as od
import DrawTree as dt

# Divides a set on a specific feature. Further can handle numeric or nominal values
def dividedata(Data, feature, threshold):
    # Make a function that tells us if a example is in the first group (true) or the second group (false)
   # Divide the Data into two sets and return them
    set1 = np.array([row for row in Data if row[feature]<threshold])
    set2 = np.array([row for row in Data if row[feature]>=threshold])
    return (set1,set2)

# Count on examples with class0,1,2

def entropy(Data):
    if len(Data) <= 1:
        return 0.0
    classes, counts = np.unique(Data[:,-1], return_counts=True) #count how many examples with class0,class1,class2
    if len(classes) <= 1:
        return 0.0  #only 1 class left, it's a leaf
    ent = 0.0
    for count in counts:
        p = float(count)/len(Data)
        if p > 1e-3:
            ent -= p*np.log2(p)
    return ent
    
class decisionnode:
  def __init__(self,feature=-1,threshold=None,Class=None,left=None,right=None):
    self.feature=feature
    self.threshold=threshold
    self.Class=Class
    self.left=left
    self.right=right

    
def buildtree(Data, k=1,feature_pool=[0,1,2,3],scoref=entropy):
    '''
    Data is the set, either whole dataset or part of it in the recursive call.
    features_pool is the candidate feature to be chosen in this node.
    scoref is the method to measure heterogeneity. By default it's entropy.
    '''
    n = len(Data)
    if n == 0: 
        return decisionnode() #len(Data) is the number of examples in a set
    current_uncertainty, best_gain = scoref(Data), 0
    last_column = len(Data[0])-1 #in implementation3, it is 4 because there are 4 data and 1 class
    
    if current_uncertainty == 0:    #this node has already been classified
        return decisionnode(Class=[Data[0,last_column],n])
    if n < k:  #this node has less than k examples and must stop ####how to classify it's label? the majority? and the number?###
        Data = Data[np.argsort(Data[:,last_column])]
        label = Data[n-1,last_column]
        return decisionnode(Class=[label,Data[:,last_column].tolist().count(label),n])

    best_feature, best_threshold = None, None
    for feature in feature_pool:
        #Data.sort(lambda x,y:cmp(x[feature],y[feature])) #a sort method in list not numpy
        Data = Data[np.argsort(Data[:,feature])]
        for i in range(1,n):
            if Data[i,last_column] != Data[i-1,last_column]:
                threshold = Data[i,feature] #set a threshold when label/class changes
                set1,set2 = dividedata(Data, feature, threshold) #define set1 and set2 as the 2 children set of a division
                p = float(len(set1))/n #p is the size of a child set relative to its parent
                #uncertainty = p*scoref(set1) + (1-p)*scoref(set2)
                gain = current_uncertainty -  p*scoref(set1) - (1-p)*scoref(set2)
                if gain > best_gain and len(set1)>0 and len(set2)>0:
                    best_gain = gain
                    best_feature, best_threshold = feature, threshold
                    best_sets = (set1, set2)
                    #print 'featrue %d, threshold %.2f, gain %.5f, N=%d to (%d and %d)' %(feature, threshold, gain, n, len(set1),len(set2))
    if best_gain > 0:
        print 'featrue %d, threshold %.1f, gain %.5f, N=%d to (%d and %d)' %(best_feature, best_threshold, best_gain, n, len(best_sets[0]),len(best_sets[1]))
        left_child = buildtree(best_sets[0],k,feature_pool,scoref)
        right_child = buildtree(best_sets[1],k,feature_pool,scoref)
        return decisionnode(best_feature,best_threshold,left=left_child,right=right_child)
    else:   #this split has no update, changing feature and threshold won't split new nodes
        Data = Data[np.argsort(Data[:,last_column])]
        label = Data[n-1,last_column]
        return decisionnode(Class=[label,'*',n])
    

def buildtree_trErr(Data, k=1, feaBagging=False,feature_pool=[0,1,2,3],scoref=entropy):
    '''
    Data is the set, either whole dataset or part of it in the recursive call.
    features_pool is the candidate feature to be chosen in this node.
    scoref is the method to measure heterogeneity. By default it's entropy.
    '''
    n = len(Data)
    if n == 0: 
        tr_err = 0
        return decisionnode(), tr_err #len(Data) is the number of examples in a set
    current_uncertainty, best_gain = scoref(Data), 0
    last_column = len(Data[0])-1 #in implementation3, it is 4 because there are 4 data and 1 class
    
    if current_uncertainty == 0:    #this node has already been classified
        tr_err = 0
        return decisionnode(Class=[Data[0,last_column],n]), tr_err
    if n < k:  #this node has less than k examples and must stop ####how to classify it's label? the majority? and the number?###
        stat = collections.Counter(Data[:,-1]).most_common()
        label = stat[0][0]
        tr_err = (len(Data) - stat[0][1])
        return decisionnode(Class=[label,Data[:,last_column].tolist().count(label),n]), tr_err

    best_feature, best_threshold = None, None
    if feaBagging == True:
        sset_feature = random.sample(feature_pool, 2)
    else:
        sset_feature = feature_pool

    for feature in sset_feature:
        #Data.sort(lambda x,y:cmp(x[feature],y[feature])) #a sort method in list not numpy
        Data = Data[np.argsort(Data[:,feature])]
        for i in range(1,n):
            if Data[i,last_column] != Data[i-1,last_column]:
                threshold = Data[i,feature] #set a threshold when label/class changes
                set1,set2 = dividedata(Data, feature, threshold) #define set1 and set2 as the 2 children set of a division
                p = float(len(set1))/n #p is the size of a child set relative to its parent
                #uncertainty = p*scoref(set1) + (1-p)*scoref(set2)
                gain = current_uncertainty -  p*scoref(set1) - (1-p)*scoref(set2)
                if gain > best_gain and len(set1)>0 and len(set2)>0:
                    best_gain = gain
                    best_feature, best_threshold = feature, threshold
                    best_sets = (set1, set2)
                    #print 'featrue %d, threshold %.2f, gain %.5f, N=%d to (%d and %d)' %(feature, threshold, gain, n, len(set1),len(set2))
    if best_gain > 0:
        print 'featrue %d, threshold %.1f, gain %.5f, N=%d to (%d and %d)' %(best_feature, best_threshold, best_gain, n, len(best_sets[0]),len(best_sets[1]))
        [left_child, lftC_err] = buildtree_trErr(best_sets[0], k, feaBagging, feature_pool, scoref)
        [right_child, rhtC_err] = buildtree_trErr(best_sets[1], k, feaBagging, feature_pool, scoref)
        tr_err = lftC_err + rhtC_err
        return decisionnode(best_feature,best_threshold,left=left_child,right=right_child), tr_err
    else:   #this split has no update, changing feature and threshold won't split new nodes
        stat = collections.Counter(Data[:,-1]).most_common()
        label = stat[0][0]
        tr_err = (len(Data) - stat[0][1])
        return decisionnode(Class=[label,'*',n]), tr_err
          

def printtree_name(tree,indent='  '):
    # Is this a leaf node?
    feature_name = ['(0)sepal length','(1)sepal width', '(2)petal length', '(3)petal width']
    class_name   = ['class 0 Iris Setosa','class 1 Iris Versicolour','class 2 Iris Virginica']
    if tree.Class!=None:
        print class_name[int(tree.Class[0])]+':',tree.Class[1],
        if len(tree.Class) > 2:
            print 'of',tree.Class[2]
        else:
            print ''
    else:
        print feature_name[tree.feature],' < '+str(tree.threshold)+' ?'
        # Print the branches
        print(indent+'T-> '),
        printtree_name(tree.left,indent+'  ')
        print(indent+'F-> '),
        printtree_name(tree.right,indent+'  ')
        
        
def printtree_index(tree,indent='  '):
    # Is this a leaf node?
    feature_name = ['(feature 0)','(feature 1)', '(feature 2)', '(feature 3)']
    class_name   = ['class 0','class 1','class 2']
    if tree.Class!=None:
        print class_name[int(tree.Class[0])]+':',tree.Class[1],
        if len(tree.Class) > 2:
            print 'of',tree.Class[2]
        else:
            print ''
    else:
        print feature_name[tree.feature]+' < '+str(tree.threshold)+' ? '
        # Print the branches
        print(indent+'T->'),
        printtree_index(tree.left,indent+'  ')
        print(indent+'F->'),
        printtree_index(tree.right,indent+'  ')

def classify(testRow,tree):
    if tree.Class != None:
        return tree.Class[0]
    else:
        value = testRow[tree.feature]
        if value < tree.threshold:
            child = tree.left
        else:
            child = tree.right
        return classify(testRow, child)

def RandomForest(trainData, testData, k, L):
    sset_size = np.int(0.8 * len(trainData))
    featrue_bagging = True

    # build forest and test
    forestList, trainError, testError = [], [], []
    for treeNum in L:
        # build forest
        print '\n*** Forest with %d trees: \n' %treeNum
        forest = []
        tr_err = 0
        for i in range(treeNum):
            sset_idx = random.sample(range(len(trainData)), sset_size)
            train_sset = trainData[sset_idx[:]]
            [tree, error] = buildtree_trErr(train_sset, k, featrue_bagging)
            forest.append(tree)
            tr_err = tr_err + error
        trainError.append((float)tr_err / treeNum)
        forestList.append(forest)

        # test on forest
        te_err = 0
        for row in testData:
            te_rst = []
            for tree in forest:
                te_rst.append(classify(row,tree))
            stat = collections.Counter(te_rst).most_common()
            label = stat[0][0]
            if np.int(label) != row[-1]:  # select majority class as test result
                te_err+= 1
        testError.append(te_err)        
    return forestList, trainError, testError

    
def RunMain():
    print '************Welcome to the World of Decision Tree!***********'
    time.clock()
    t0 = float(time.clock())
    # # load data, and save data.
    [train_data, test_data] = ld.LoadData()
    t1 = float(time.clock())
    print '[done] Loading data File. using time %.4f s, \n' % (t1-t0)
    t0 = t1
    saveDir    = "./iris-Result/"
    
    
    #******************Part 1***************************
    print '******Part 1*****'
    k_max = 25
    error_train, error_test = np.zeros(k_max+1), np.zeros(k_max+1)

    
    trees = []          #######delete if no sorage of trees
    for k in range(1,k_max+1):
        print '\n****stop when number in node is less than: %d   *****' %(k)
        tree=buildtree(train_data,k)
        # [tree, tr_err] = buildtree_trErr(train_data, k)
        # error_train[k] = tr_err
        printtree_index(tree)
        dt.drawtree(tree,saveDir+'tree_k=%d.jpg' %(k))
        trees.append(tree)          #######delete if no sorage of trees

    t1 = float(time.clock())
    print '[done] Learn %d trees. using time %.4f s, \n' % (k_max,t1-t0)
    t0 = t1
    for k in range(1,k_max+1):  #######delete if no sorage of trees
        #training errors
        for row in train_data:
            if classify(row,trees[k-1]) != row[-1]:  #######delete's[k-1]' if no sorage of trees
                error_train[k] += 1
        #testing errors
        for row in test_data:
            if classify(row,trees[k-1]) != row[-1]:   #######delete's[k-1]' if no sorage of trees
                error_test[k] += 1
    '''
    # b---blue   c---cyan  g---green    k----black
    # m---magenta r---red  w---white    y----yellow
    '''
    error_train[:] = 1/float(len(train_data))*error_train
    error_test[:] = 1/float(len(test_data))*error_test
    plt.figure(1)
    plt.plot(error_train, 'r',label="classify learning data")
    plt.plot(error_test, 'b',label="classify testing data")
    plt.xlim(0, k_max)
    plt.ylim(0, 0.5)
    plt.xlabel('k')
    plt.ylabel('error percentage')  
    plt.legend(loc='upper right')
    plt.grid(True)  
    plt.savefig(saveDir+"Part1_error vs k=0-%d" %(k_max))
    #printtree_name(tree)
    #s1, s2 = dividedata(tmp,0,5.5)
    t1 = float(time.clock())
    print '[done] test trainning data and testing data. using time %.4f s.\n' % (t1-t0)
    t0 = t1
    
    
    
    #******************Part 2***************************
    # f = [0, 1, 2, 3]
    # feature_pool = random.sample(f, 2)  #choose 2/4 features out of 4
    # feature_pool.sort()
    pdb.set_trace()
    L = [5, 10, 15, 20, 25, 30]
    fTrain_err, fTest_err = np.zeros((k_max, len(L))), np.zeros((k_max, len(L)))
    for k in range(1, k_max+1):
        [forest, tr_err, te_err]=RandomForest(train_data, test_data, k, L)
        fTrain_err[k-1, :] = tr_err[:]  # number of error example
        fTest_err[k-1, :] = tr_err[:]  # number of error example

    # plot


if __name__ == "__main__":
    RunMain()
