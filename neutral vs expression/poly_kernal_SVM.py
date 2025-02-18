import numpy as np
from cvxopt import matrix, solvers
from dataloader_kernal import DataLoader
import scipy.io as sc
from PCA_reduction import PCA_compressed_train, PCA_compressed_test
from MDA_reduction import MDA_train, MDA_test

def poly_kernal(x,y, r = 2):
    return (x@y.T + 1)**r


def poly_SVM(train_data, train_label):
    
    train_data = train_data.T # each row is one data
    K = np.zeros((train_data.shape[0], train_data.shape[0]))
    m, n = train_data.shape
    for i in range(train_data.shape[0]):
        for j in range(train_data.shape[0]):
            K[i][j] = poly_kernal(train_data[i], train_data[j])
    
    train_label = train_label.reshape((-1,1))
    
    P = matrix(K * (train_label@train_label.T))
    q = matrix(np.ones((m,1))*-1)
    A = matrix(train_label.reshape((1,-1)))
    b = matrix(np.zeros((1,1)))
    G = matrix(np.identity(m)*-1)
    h = matrix(np.zeros((m,1)))
    
    solution = solvers.qp(P, q, G, h, A, b)
    mus = np.array(solution['x'])
    index = (mus > 1e-4).flatten()
    # print(mus)
    SV = train_data[index]
    SV_y = train_label[index]
    
    theta_0 = 0
    
    for i in range(SV.shape[0]):
        theta_0 += SV_y[i] - poly_kernal(SV[i], SV[i])*SV_y[i]*mus[i]
        
    theta_0 = theta_0 / SV.shape[0]
    
    return SV, SV_y, mus, theta_0

def test(test_data, test_label, SV, SV_y, mus, theta_0):
    
    n_sample = 0
    n_correct = 0
    
    for i in range(test_data.shape[1]):
        image = test_data[:,i].reshape((1,-1))
        sum = 0
        n_sample += 1
        for j in range(SV.shape[0]):
            sum += poly_kernal(image, SV[j])*SV_y[j]*mus[j]
        sum = sum + theta_0
        y_hat = np.sign(sum)
        if y_hat == test_label[i]:
            n_correct += 1
            
        
    return round(n_correct*100/n_sample,2)        
    
def main():
    data = sc.loadmat('data.mat')
    face = data['face']
    face = face.reshape((-1, face.shape[2]))
    train_data, train_label, test_data, test_label = DataLoader(face)
    train_data, mean, reduced_eig_vect,_ = PCA_compressed_train(train_data)
    test_data = PCA_compressed_test(test_data, mean, reduced_eig_vect)
    SV, SV_y, mus, theta_0 = poly_SVM(train_data, train_label)
    test_error = test(test_data, test_label, SV, SV_y, mus, theta_0)
    print('testing error in PCA data: ' + str(test_error))

    train_data, train_label, test_data, test_label = DataLoader(face)
    train_data, direction,_ = MDA_train(train_data)
    test_data = MDA_test(test_data, direction)
    SV, SV_y, mus, theta_0 = poly_SVM(train_data, train_label)
    test_error = test(test_data, test_label, SV, SV_y, mus, theta_0)
    print('testing error in MDA data: ' + str(test_error))
    
    
if __name__ == '__main__':
    main()


