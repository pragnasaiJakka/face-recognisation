from re import X
import scipy.io as sc
import numpy as np
import matplotlib.pyplot as plt

def DataLoader(face):
    
    for i in range(0, face.shape[1]-2, 3):
        
        if i == 0:
            class_1 = face[:,i].reshape((face.shape[0],1))
            class_2 = face[:,i+1].reshape((face.shape[0],1))
            continue
        
        class_1 = np.concatenate((class_1, face[:,i].reshape((face.shape[0],1))),axis = 1)
        class_2 = np.concatenate((class_2, face[:,i+1].reshape((face.shape[0],1))),axis = 1)
        

    # class_1 = class_1[:,np.random.permutation(class_1.shape[1])]
    # class_2 = class_2[:,np.random.permutation(class_2.shape[1])]
    
    class_1 = np.vstack((class_1, -np.ones((1, class_1.shape[1]))))
    class_2 = np.vstack((class_2, np.ones((1, class_2.shape[1]))))
    
    class_1_train = class_1[:,:int(0.9*class_1.shape[1])]
    class_1_test = class_1[:,int(0.9*class_1.shape[1]):]
    class_2_train = class_2[:, :int(0.9*class_2.shape[1])]
    class_2_test = class_2[:, int(0.9*class_2.shape[1]):]
    
    test_set = np.hstack((class_1_test, class_2_test))
    train_set = np.hstack((class_1_train, class_2_train))

    # test_set = test_set[:, np.random.permutation(test_set.shape[1])]
    
    test_data  = test_set[:-1,:]
    test_label = test_set[-1,:]
    train_data = train_set[:-1,:]
    train_label = train_set[-1,:]
    
    return train_data, train_label, test_data, test_label
    
def main():
    data = sc.loadmat('data.mat')
    face = data ['face']
    face = face.reshape((-1,face.shape[2]))
    _,_,_,_ = DataLoader(face)
    
    
if __name__ == '__main__':
    main()