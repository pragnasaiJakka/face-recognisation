from tkinter import Y
import numpy as np
from dataloader import DataLoader
import scipy.io as sc
from tqdm import tqdm
from PCA_reduction import PCA_compressed_train, PCA_compressed_test
from MDA_reduction import MDA_test, MDA_train

def max_repeat_neighbour(arr):

    count = {}
    freq = [[] for i in range(len(arr) + 1)]

    for n in arr:
        count[n] = 1 + count.get(n,0)
    
    for n, c in count.items():
        freq[c].append(n)
    
    for i in range(len(freq)-1,0,-1):
        for n in freq[i]:
            res = n
            if res != 0:
                return res


def KNN(x_train, y_train, x_test, y_test, k_neighbor):
    
    #shuffle training data and training label
    y_train = np.array(y_train).reshape((1,len(y_train)))
    training_data = np.concatenate((x_train, y_train),axis=0)
    training_data = training_data[:,np.random.permutation(training_data.shape[1])]
    x_train, y_train = training_data[:-1,:], training_data[-1,:]
    
    
    #shuffle testing data and testing label
    y_test = np.array(y_test).reshape((1,len(y_test)))
    testing_data = np.concatenate((x_test, y_test))
    testing_data = testing_data[:,np.random.permutation(testing_data.shape[1])]
    x_test, y_test = testing_data[:-1,:], testing_data[-1,:]
    
    n_sample = 0 # total number of testing data
    n_correct = 0 # number of testing data classified correctly
    
    for i in tqdm(range(x_test.shape[1])):

        euclidean_dist_arr = []
        x_i = x_test[:,i].reshape((x_test.shape[0],1))
        
        for j in range(x_train.shape[1]):
            x_j = x_train[:,j].reshape((x_train.shape[0],1))
            euclidean_dist = ((x_i - x_j).T@(x_i - x_j)).item()
            euclidean_dist_arr.append(euclidean_dist)
        
        idx = np.argsort(euclidean_dist_arr)
        
        y_train_sort = y_train[idx]
        
        y_hat = max_repeat_neighbour(y_train_sort[:k_neighbor])
        
        if y_hat == 0:
            y_hat = -1 
        
        n_sample += 1
        
        if y_hat == y_test[i]:
            n_correct += 1

    test_accuracy = n_correct*100/n_sample
    
    return test_accuracy

def main():
    
    data = sc.loadmat('data.mat')
    face = data['face']
    face = face.reshape((-1,face.shape[2]))
    k_neighbor = 5
    #PCA compressed
    x_train, y_train, x_test, y_test = DataLoader(face)
    x_train, mean_face, reduced_eig_vect,_ = PCA_compressed_train(x_train)
    x_test = PCA_compressed_test(x_test, mean_face, reduced_eig_vect)
    test_accuracy = KNN(x_train, y_train, x_test, y_test,k_neighbor)
    print('test accuracy in PCA')
    print(test_accuracy )
    
    #MDA compressed
    x_train, y_train, x_test, y_test= DataLoader(face)
    x_train, direction,_ = MDA_train(x_train)
    x_test = MDA_test(x_test, direction)
    test_accuracy = KNN(x_train, y_train, x_test, y_test,k_neighbor)
    print('test accuracy in MDA')
    print(test_accuracy)

if __name__ == '__main__':
    main()
