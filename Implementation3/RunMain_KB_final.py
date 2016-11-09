"""
Created on Fri Nov 04 23:01:07 2016

@author: Kaibo Liu
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import time
import random
import pdb
import copy

import LoadData   as ld
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


def buildtree(Data,k=1,n_feature=4,scoref=entropy):
    '''
    Data is the set, either whole dataset or part of it in the recursive call.
    features_pool is the candidate feature to be chosen in this node.
    scoref is the method to measure heterogeneity. By default it's entropy.
    '''
    n = len(Data)
    if n == 0:
        return decisionnode() #len(Data) is the number of examples in a set

    feature_pool = [0,1,2,3]
    if n_feature <4:
        feature_pool = np.random.choice(feature_pool,n_feature,replace=False)
        #need to swap this 2 features if f[0]>f[1]
        #feature_pool = random.sample(f, 2)

    current_uncertainty, best_gain = scoref(Data), 0
    #last_column = len(Data[0])-1 #in implementation3, it is 4 because there are 4 data and 1 class

    if current_uncertainty == 0:    #this node has already been classified
        return decisionnode(Class=[Data[0,-1],n])
    if n < k:  #this node has less than k examples and must stop ####how to classify it's label? the majority? and the number?###
        #Data = Data[np.argsort(Data[:,-1])]
        #label = Data[n-1,-1]
        label = mode(Data[:,-1])
        return decisionnode(Class=[label[0][0],label[1][0],n])

    best_feature, best_threshold = None, None
    for feature in feature_pool:
        #Data.sort(lambda x,y:cmp(x[feature],y[feature])) #a sort method in list not numpy
        Data = Data[np.argsort(Data[:,feature])]
        for i in range(1,n):
            if Data[i,-1] != Data[i-1,-1]:
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
<<<<<<< HEAD
        ######## output the threshold and gain ########
=======
>>>>>>> 2cccb4349ad51fd23d2917a978c23be5b88c31f2
        #print 'featrue %d, threshold %.1f, gain %.5f, N=%d to (%d and %d)' %(best_feature, best_threshold, best_gain, n, len(best_sets[0]),len(best_sets[1]))
        left_child = buildtree(best_sets[0],k,n_feature,scoref)
        right_child = buildtree(best_sets[1],k,n_feature,scoref)
        return decisionnode(best_feature,best_threshold,left=left_child,right=right_child)
    else:   #this split has no update, changing feature and threshold won't split new nodes
        '''Data = Data[np.argsort(Data[:,last_column])]
        label = Data[n-1,last_column]'''
        label = mode(Data[:,-1])
        return decisionnode(Class=[label[0][0],label[1][0],n])


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


def run_part(data_train,data_test,k_max=1,saveDir="./iris-Result/",rate_subset=1,L=1,show_tree=True):
    t0 = float(time.clock())
    #k_p = [1,4,7,9,38,46,81]
    n_subset = int(rate_subset* len(data_train))
    error_train, error_test = np.zeros(k_max+1), np.zeros(k_max+1)
    #learn a tree with k as stop condition
    for k in range(1,k_max+1):
    #for k in k_p:
        predict_matrix_train, predict_matrix_test = [],[]
        for l in range(L): #creat L trees for one k
            #print '\n****tree_%d of L=%d_k=%d*****' %(l+1,L,k)
            if saveDir[-2] == '1':
                tree = buildtree(data_train,k)
            else:          #if it's problem2, set the number of features: all or random 2
                n_feature = 2 #choose 2 features out of 4 in buildtree function
                ##if it's problem2, and no matter if subset n<N, choose random subset
                index = np.array(range(len(data_train)))
                subset = data_train[np.random.choice(index,n_subset,replace=True)]
                tree = buildtree(subset,k,n_feature)

            if show_tree:
                printtree_index(tree)
                if saveDir[-2] == '1':
                    dt.drawtree(tree,saveDir+'tree_k=%d.jpg' %(k))
                else:
                    dt.drawtree(tree,saveDir+'tree_%d of L=%d_k=%d.jpg' %(l+1,L,k))
            #trees.append(tree)          #######delete if no sorage of trees

            predict_1tree_train, predict_1tree_test = [],[]
            #training errors with this tree l of L
            for row in data_train:
                predict_1tree_train.append(classify(row,tree))  #1*N_train
            #testing errors with this tree l of L
            for row in data_test:
                predict_1tree_test.append(classify(row,tree))     #1*N_test
            predict_matrix_train.append(np.array(predict_1tree_train))      #finally L*N_train
            predict_matrix_test.append(np.array(predict_1tree_test))      #finally L*N_test

        '''test training and testing data w.r.t k
        vote the majority of predicted class for every example'''
        predict_matrix_train = np.transpose(predict_matrix_train)
        predict_matrix_test = np.transpose(predict_matrix_test)
        for i in range(len(data_train)):
            if mode(predict_matrix_train[i])[0][0] != data_train[i,-1]:
                error_train[k] += 1
        for i in range(len(data_test)):
            if mode(predict_matrix_test[i])[0][0] != data_test[i,-1]:
                error_test[k] += 1

    accuracy_train = 100*(1-1/float(len(data_train))*error_train)
    accuracy_test = 100*(1-1/float(len(data_test))*error_test)
    t1 = float(time.clock())
    print '[done] Learn and test k%d*L%d trees. using time %.4f s, \n' % (k_max,L,t1-t0)
    t0 = t1

    '''b---blue   c---cyan  g---green    k----black
    m---magenta r---red  w---white    y----yellow'''
    if saveDir[-2] == '1':
        plt.figure(1,figsize=(8, 6), dpi=80)
        plt.plot(range(1,k_max+1),accuracy_train[1:], 'r',label="training data")
        plt.plot(range(1,k_max+1),accuracy_test[1:], 'b',label="testing data")
        plt.xlim(1, k_max)
        plt.ylim(65, 100)
        plt.xlabel('k')
<<<<<<< HEAD
        plt.ylabel('accuracy (%)')  
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(saveDir+"accuracy vs k=0-%d" %(k_max))
    elif saveDir[-2] == '2':
        return accuracy_train,accuracy_test
    
        
=======
        plt.ylabel('accuracy percentage')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(saveDir+"accuracy vs k=0-%d" %(k_max))
    elif saveDir[-2] == '2':
        return error_train,error_test


>>>>>>> 2cccb4349ad51fd23d2917a978c23be5b88c31f2
def get_colour(color):
    '''
    b---blue   c---cyan  g---green    k----black
    m---magenta r---red  w---white    y----yellow'''
    color_set = ['r','b','m','g','c','k','y']
    return color_set[color % 7]

def RunMain():
    print '************Welcome to the World of Decision Tree!***********'
    time.clock()
    t0 = float(time.clock())
    # # load data, and save data.
    [data_train, data_test] = ld.LoadData()
    t1 = float(time.clock())
    print '[done] Loading data File. using time %.4f s, \n' % (t1-t0)
    t0 = t1
    saveDir    = "./iris-Result-3/"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    #******************Part 1***************************
    print '******Part 1*****'
    k_max = 50
<<<<<<< HEAD
    #run_part(data_train,data_test,k_max,saveDir+'Part1_',rate_subset=1,L=1,show_tree=False)
    
    #******************Part 2***************************
    print '******Part 2*****'
    rate_subset = 1
    Loop_list = [5,10,15,20,25,30]
    #Loop_list = [1,1,1,1,1,5]
    random_runs = 10
    colour = 0
    for l in Loop_list:
        accuracy_train_10,accuracy_test_10 = np.zeros(k_max+1), np.zeros(k_max+1)
        for iteration in range(random_runs): #10 randm runs
            accuracy_train,accuracy_test =run_part(data_train,data_test,k_max,saveDir+'Part2_',rate_subset,l,show_tree=False)
            accuracy_train_10,accuracy_test_10 = accuracy_train_10 + accuracy_train,accuracy_test_10+accuracy_test
        
        accuracy_train = 1/float(random_runs)*accuracy_train_10
        accuracy_test = 1/float(random_runs)*accuracy_test_10
        #training data VS k
        plt.figure(2,figsize=(8, 5), dpi=80)#, facecolor='w', edgecolor='k')
        plt.plot(range(1,k_max+1),accuracy_train[1:], get_colour(colour),label="training data, L=%d" %(l))
        plt.xlim(1, k_max)
        plt.ylim(85, 100)
=======
    #run_part(data_train,data_test,k_max,saveDir+'Part1_',rate_subset=1,L=1)

    #******************Part 2***************************
    print '******Part 2*****'
    rate_subset = 0.9
    Loop_list = [5,10,15,20,25,30]
    #Loop_list = [1,1,1,1,1,5]
    random_runs = 1
    colour = 0
    for l in Loop_list:
        print l
        error_train_10,error_test_10 = np.zeros(k_max+1), np.zeros(k_max+1)
        for iteration in range(random_runs): #10 randm runs
            error_train,error_test =run_part(data_train,data_test,k_max,saveDir+'Part2_',rate_subset,l,show_tree=False)
            error_train_10,error_test_10 = error_train_10 + error_train,error_test_10+error_test

        #training data VS k
        plt.figure(2,figsize=(8, 5), dpi=80)#, facecolor='w', edgecolor='k')
        plt.plot(range(1,k_max+1),100/float(random_runs)*error_train_10[1:], get_colour(colour),label="learning data, L=%d" %(l))
        plt.xlim(1, k_max)
        plt.ylim(90, 101)
>>>>>>> 2cccb4349ad51fd23d2917a978c23be5b88c31f2
        plt.xlabel('k')
        plt.ylabel('accuracy (%)')
        plt.legend(loc='lower left')
        plt.grid(True)

        #testing data VS k
        plt.figure(3,figsize=(8, 5), dpi=80)#, facecolor='w', edgecolor='k')
<<<<<<< HEAD
        plt.plot(range(1,k_max+1),accuracy_test[1:], get_colour(colour)+'--',label="testing data, L=%d" %(l))
        plt.xlim(1, k_max)
        plt.ylim(85, 100)
=======
        plt.plot(range(1,k_max+1),100/float(random_runs)*error_test_10[1:], get_colour(colour)+'--',label="testing data, L=%d" %(l))
        plt.xlim(1, k_max)
        plt.ylim(70, 101)
>>>>>>> 2cccb4349ad51fd23d2917a978c23be5b88c31f2
        plt.xlabel('k')
        plt.ylabel('accuracy (%)')
        plt.legend(loc='lower left')
        plt.grid(True)
<<<<<<< HEAD
        
        #training & testing with all Ls + upper right lengend
        plt.figure(4,figsize=(8, 5), dpi=80)#, facecolor='w', edgecolor='k')
        plt.plot(range(1,k_max+1),accuracy_train[1:], get_colour(colour),label="training data, L=%d" %(l))
        plt.plot(range(1,k_max+1),accuracy_test[1:], get_colour(colour)+'--',label="testing data, L=%d" %(l))
        plt.xlim(1, k_max*1.65)
        plt.ylim(90, 100)
=======

        #training & testing with only all Ls + upper right lengend
        plt.figure(4,figsize=(8, 5), dpi=80)#, facecolor='w', edgecolor='k')
        plt.plot(range(1,k_max+1),100/float(random_runs)*error_train_10[1:], get_colour(c),label="learning data, L=%d" %(l))
        plt.plot(range(1,k_max+1),100/float(random_runs)*error_test_10[1:], get_colour(colour)+'--',label="testing data, L=%d" %(l))
        plt.xlim(1, k_max*1.65)
        plt.ylim(80, 101)
>>>>>>> 2cccb4349ad51fd23d2917a978c23be5b88c31f2
        plt.xlabel('k')
        plt.ylabel('accuracy (%)')
        plt.legend(loc='upper right')
        plt.grid(True)
<<<<<<< HEAD
        
        #training & testing with all Ls + lower left lengend
        plt.figure(5,figsize=(8, 5), dpi=80)#, facecolor='w', edgecolor='k')
        plt.plot(range(1,k_max+1),accuracy_train[1:], get_colour(colour),label="training data, L=%d" %(l))
        plt.plot(range(1,k_max+1),accuracy_test[1:], get_colour(colour)+'--',label="testing data, L=%d" %(l))
        plt.xlim(1, k_max)
        plt.ylim(50, 100)
=======

        #training & testing with only all Ls + lower left lengend
        plt.figure(5,figsize=(8, 5), dpi=80)#, facecolor='w', edgecolor='k')
        plt.plot(range(1,k_max+1),100/float(random_runs)*error_train_10[1:], get_colour(c),label="learning data, L=%d" %(l))
        plt.plot(range(1,k_max+1),100/float(random_runs)*error_test_10[1:], get_colour(colour)+'--',label="testing data, L=%d" %(l))
        plt.xlim(1, k_max)
        plt.ylim(60, 101)
>>>>>>> 2cccb4349ad51fd23d2917a978c23be5b88c31f2
        plt.xlabel('k')
        plt.ylabel('accuracy (%)')
        plt.legend(loc='lower left')
        plt.grid(True)

        #training & testing with only one L
        plt.figure(l+6,figsize=(8, 5), dpi=80)#, facecolor='w', edgecolor='k')
<<<<<<< HEAD
        plt.plot(range(1,k_max+1),accuracy_train[1:], get_colour(colour),label="training data, L=%d" %(l))
        plt.plot(range(1,k_max+1),accuracy_test[1:], get_colour(colour)+'--',label="testing data, L=%d" %(l))
        plt.xlim(1, k_max)
        plt.ylim(90, 100)
=======
        plt.plot(range(1,k_max+1),100/float(random_runs)*error_train_10[1:], get_colour(c),label="learning data, L=%d" %(l))
        plt.plot(range(1,k_max+1),100/float(random_runs)*error_test_10[1:], get_colour(colour)+'--',label="testing data, L=%d" %(l))
        plt.xlim(1, k_max)
        plt.ylim(80, 101)
>>>>>>> 2cccb4349ad51fd23d2917a978c23be5b88c31f2
        plt.xlabel('k')
        plt.ylabel('accuracy (%)')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(saveDir+'Part2_L%d_trees_accuracy vs k%d' %(l,k_max))

        colour += 1
<<<<<<< HEAD
        
=======


>>>>>>> 2cccb4349ad51fd23d2917a978c23be5b88c31f2
    plt.figure(2)
    plt.savefig(saveDir+'Part2_all_%dL_trees_training_accuracy vs k%d' %(len(Loop_list),k_max))
    plt.figure(3)
<<<<<<< HEAD
    plt.savefig(saveDir+'Part2_all_%dL_trees_testing_accuracy vs k%d' %(len(Loop_list),k_max))    
=======
    plt.savefig(saveDir+'Part2_%dXL_trees_testing_accuracy vs k%d' %(len(Loop_list),k_max))
>>>>>>> 2cccb4349ad51fd23d2917a978c23be5b88c31f2
    plt.figure(4)
    plt.savefig(saveDir+'Part2_all_%dL_trees_all_accuracy vs k%d_upper_right' %(len(Loop_list),k_max))
    plt.figure(5)
    plt.savefig(saveDir+'Part2_all_%dL_trees_all_accuracy vs k%d_lower_left' %(len(Loop_list),k_max))

if __name__ == "__main__":
    RunMain()
