from random import randint
import numpy as np
from dataloader import DataLoader
from PCA_reduction import PCA_compressed_train, PCA_compressed_test
from MDA_reduction import MDA_train, MDA_test
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io as sc

def bayes(x_train, y_train, x_test, y_test):

    def findCovarinace(x_train):
    
        n_sample = int(x_train.shape[1] / 2)
        
        for i in range(0,x_train.shape[1] - (n_sample - 1),n_sample):
            if i == 0:
                class_mean_arr = np.mean((x_train[:,i:i+n_sample]),axis = 1).reshape((x_train.shape[0],1))
                continue
            class_mean_arr = np.concatenate((class_mean_arr, np.mean((x_train[:,i:i+n_sample]),axis=1).reshape(x_train.shape[0],1)),axis=1)

        for i in range(0,x_train.shape[1] - (n_sample - 1),n_sample):
            
            if i == 0 : 
            
                each_class_mean = class_mean_arr[:,int(i/n_sample)].reshape((x_train.shape[0],1))
                class_i = x_train[:,i:i+n_sample]
                mean_centered_class = class_i - each_class_mean
                covariance_i = (mean_centered_class @ mean_centered_class.T)/n_sample
                if np.linalg.det(covariance_i) == 0:
                    covariance_i = covariance_i + 0.5*np.identity(covariance_i.shape[0])
                covariance_matrix = covariance_i
                continue
            
            each_class_mean = class_mean_arr[:,int(i/n_sample)].reshape((x_train.shape[0],1))
            class_i = x_train[:,i:i+n_sample]
            mean_centered_class = class_i - each_class_mean
            covariance_i = (mean_centered_class@mean_centered_class.T)/n_sample
            if np.linalg.det(covariance_i) == 0:
                    covariance_i = covariance_i + 0.5*np.identity(covariance_i.shape[0])
            covariance_matrix = np.dstack((covariance_matrix,covariance_i))     
            
        return class_mean_arr, covariance_matrix
    
    def train(x_train, y_train, class_mean_arr, covariance_matrix):
        
        # x_train = (504,400) array with each column is flattened image -> 400 train Image with 200 class
        # y_train = (400,) with each element is a class of each x_train column
        # class_mean_arr = (504,200) with each column is a mean of each class
        # covariance_matrix = (504,504,200) with each depth element is (504,504) covariance matrix of each class
        
        k = x_train.shape[0] # data dimension
        n_sample = 0 # total training sample
        n_correct = 0 # total correct classified sample during training
        for i in tqdm(range(x_train.shape[1])):
            # print(i)
            likelihood_arr = [] # array to store likelihood of each class
            image = x_train[:,i].reshape((x_train.shape[0],1))
            
            for j in range(covariance_matrix.shape[2]):
                # take the mean of each class one by one 1,2,3...
                class_mean = class_mean_arr[:,j].reshape((x_train.shape[0],1))
                # subtract the mean of each class from training sample
                x = image - class_mean
                # take the covariance matrix of each class one by one 1,2,3...
                covariance_matrix_j = covariance_matrix[:,:,j].reshape((covariance_matrix.shape[0],covariance_matrix.shape[1]))
                # find the liklihood of the each class for given image
                likelihood = -(k/2)*np.log(2*np.pi)-(1/2)*np.linalg.det(covariance_matrix_j) - (1/2)*(x.T @ np.linalg.inv(covariance_matrix_j) @ x)
                likelihood_arr.append(likelihood)
            
            # find the class whose likelihood is maximum   
            max_index = likelihood_arr.index(max(likelihood_arr))
            n_sample += 1
            # if the maximum likelihood class label match with the ground truthe label, count it as correct classification. 
            if max_index == y_train[i]:
                n_correct += 1       
        
        train_error = round(n_correct*100/n_sample, 2)
        
        return train_error
    
    def test(x_test, y_test, class_mean_arr, covariance_matrix):
        
        n_sample = 0 # total testing sample
        n_correct = 0 # Number of correctly classified sample
        k = len(x_test) # data dimension
        
        for i in tqdm(range(x_test.shape[1])):
            likelihood_arr = []
            image = x_test[:,i].reshape((x_test.shape[0],1))
            
            for j in range(covariance_matrix.shape[2]):
                class_mean = class_mean_arr[:,j].reshape((x_test.shape[0],1))
                x = image - class_mean
                covariance_matrix_j = covariance_matrix[:,:,j].reshape((covariance_matrix.shape[0],covariance_matrix.shape[1]))
                likelihood = -(k/2)*np.log(2*np.pi)-(1/2)*np.linalg.det(covariance_matrix_j)-(1/2)*(x.T@np.linalg.inv(covariance_matrix_j)@x)
                likelihood_arr.append(likelihood)
        
            max_index = likelihood_arr.index(max(likelihood_arr))
            n_sample += 1
            if max_index == y_test[i]:
                n_correct += 1
                
        test_accuracy = round(n_correct*100/n_sample,2)
        
        return test_accuracy
                
    class_mean_arr, covariance_matrix = findCovarinace(x_train)
    train_error = train(x_train, y_train, class_mean_arr, covariance_matrix)
    test_error = test(x_test, y_test, class_mean_arr, covariance_matrix)
    
    return train_error , test_error
        

def main():
    data = sc.loadmat('data.mat')
    face = data ['face']
    face = face.reshape((-1,face.shape[2]))
    
    #PCA analysis
    x_train, y_train, x_test, y_test = DataLoader(face)
    x_train, mean_face, reduced_eig_vect,_ = PCA_compressed_train(x_train)
    x_test = PCA_compressed_test(x_test, mean_face, reduced_eig_vect)
    train_error , test_error = bayes(x_train, y_train, x_test, y_test)
    print('Test Error in PCA')
    print(test_error)
    
    #MDA analysis
    x_train, y_train, x_test, y_test = DataLoader(face)
    x_train, reduced_eig_vect,_ = MDA_train(x_train)
    x_test = MDA_test(x_test, reduced_eig_vect)
    train_error , test_error = bayes(x_train, y_train, x_test, y_test)
    print('Test Error in MDA')
    print(test_error)

if __name__ == '__main__':
    main()