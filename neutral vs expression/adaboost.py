import numpy as np
from cvxopt import matrix, solvers
from dataloader_kernal import DataLoader
import scipy.io as sc
from PCA_reduction import PCA_compressed_train, PCA_compressed_test
from MDA_reduction import MDA_train, MDA_test
import matplotlib.pyplot as plt

def linear_kernal(x,y):
    return x@y.T

def linear_SVM(train_data, train_label):
    
    train_data = train_data.T # each row is one data
    K = np.zeros((train_data.shape[0], train_data.shape[0]))
    m, n = train_data.shape
    for i in range(train_data.shape[0]):
        for j in range(train_data.shape[0]):
            K[i][j] = linear_kernal(train_data[i], train_data[j])
    
    train_label = train_label.reshape((-1,1))
    
    P = matrix(K * (train_label@train_label.T))
    q = matrix(np.ones((m,1))*-1)
    A = matrix(train_label.reshape((1,-1)))
    b = matrix(np.zeros((1,1)))
    G = matrix(np.identity(m)*-1)
    h = matrix(np.zeros((m,1)))
    
    solution = solvers.qp(P, q, G, h, A, b)
    mus = np.array(solution['x'])
    # print(mus)
    index = (mus > 1e-5).flatten()
    
    SV = train_data[index]
    SV_y = train_label[index]
    mus = mus[index]

    # SV = train_data
    # SV_y = train_label
    # mus = mus
    
    theta_0 = 0
    
    for i in range(SV.shape[0]):
        theta_0 += SV_y[i] - linear_kernal(SV[i], SV[i])*SV_y[i]*mus[i]
        
    theta_0 = theta_0 / SV.shape[0]
    
    return SV, SV_y, mus, theta_0

def test(a_n_arr, phi_matrix, test_data, test_label):
    
    n_correct = 0
    n_sample = 0
    
    for i in range(test_data.shape[1]):
        image = np.matrix(test_data[:,i])
        
        sum = 0
        
        for j in range(len(a_n_arr)):
            a = a_n_arr[j]
            phi = phi_matrix[j]
            SV, SV_y, mu, theta_0 = phi[0], phi[1], phi[2], phi[3]
            for k in range(SV.shape[0]):
                sum += linear_kernal(image, SV[k])*SV_y[k]*mu[k]*a
                
            sum = sum + theta_0*a
                
        y_hat = np.sign(sum)
        n_sample += 1
        if y_hat == test_label[i]:
            n_correct += 1
            
    return round(n_correct*100/n_sample,2)

def adaboost_SVM(train_data, train_label, test_data, test_label, reduction_method, n_iteration=8):
    
    w_n_arr = np.ones(train_data.shape[1])
    
    a_n_arr = []
    
    phi_matrix = []
    
    for n in range(n_iteration):
        
        P_n_arr = w_n_arr / np.sum(w_n_arr)
    
        a = 0
        
        epsilon = 0
        
        class_1_data = train_data[:,:int(train_data.shape[1]/2)]
        class_2_data = train_data[:,int(train_data.shape[1]/2):]
        class_1_data = class_1_data[:,np.random.permutation(class_1_data.shape[1])]
        class_2_data = class_2_data[:,np.random.permutation(class_2_data.shape[1])]
        class_1_data = class_1_data[:,:120]
        class_2_data = class_2_data[:,:120]
        train_data_modify = np.hstack((class_1_data, class_2_data))
        
        
        class_1_label = train_label[:120]
        class_2_label = train_label[-120:]
        train_label_modify = np.hstack((class_1_label, class_2_label))

        SV, SV_y, mus, theta_0 = linear_SVM(train_data,train_label)
        
        phi_matrix.append([SV, SV_y, mus, theta_0])
        
        sum_arr = []
        n_incorrect = 0
        
        for i in range(train_data.shape[1]):
            image = train_data[:,i].reshape((1,-1))
            sum = 0
            for j in range(SV.shape[0]):
                
                sum += linear_kernal(image, SV[j])*SV_y[j]*mus[j]
                
            sum = sum + theta_0
            sum_arr.append(sum)
            y_hat = np.sign(sum)
            if y_hat != train_label[i]:
                n_incorrect += 1
                epsilon += P_n_arr[i]

        if epsilon  != 0 :
            a = (1/2)*np.log((1-epsilon)/epsilon)
        else:
            a = 0
            
        a_n_arr.append(a)

        for i in range(len(w_n_arr)):
            w_n_arr[i] = w_n_arr[i] * np.exp(-train_label[i]*a*sum_arr[i])
        
        
        n_correct = 0
        n_sample = 0
    
        for i in range(train_data.shape[1]):
            image = np.matrix(train_data[:,i])
        
            sum = 0
        
            for j in range(len(a_n_arr)):
                a = a_n_arr[j]
                phi = phi_matrix[j]
                SV, SV_y, mu, theta_0 = phi[0], phi[1], phi[2], phi[3]
                for k in range(SV.shape[0]):
                    sum += linear_kernal(image, SV[k])*SV_y[k]*mu[k]*a
                
            sum = sum + theta_0*a
                
            y_hat = np.sign(sum)
            n_sample += 1
            if y_hat == train_label[i]:
                n_correct += 1
        
        test_error = test(a_n_arr, phi_matrix, test_data, test_label )
        
        print('For' + ' ' + str(reduction_method) + ' '+ 'test accuracy after' + ' ' + str(n) + ' '+'number of iteration' + ' : ' + str(test_error))
    
    
def main():
    
    data = sc.loadmat('data.mat')
    face = data['face']
    face = face.reshape((-1, face.shape[2]))
    
    train_data, train_label, test_data, test_label = DataLoader(face)
    train_data,mean, reduced_eig_vect,_ = PCA_compressed_train(train_data)
    test_data = PCA_compressed_test(test_data, mean, reduced_eig_vect)
    adaboost_SVM(train_data, train_label, test_data, test_label, 'PCA')
    

    train_data, train_label, test_data, test_label = DataLoader(face)
    train_data, direction,_ = MDA_train(train_data)
    test_data = MDA_test(test_data, direction)
    adaboost_SVM(train_data, train_label, test_data, test_label, 'MDA')
    
    
if __name__ == '__main__':
    main()


