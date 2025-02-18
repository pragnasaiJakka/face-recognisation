from re import X
import scipy.io as sc
import numpy as np
import matplotlib.pyplot as plt
# from PCA_reduction import PCA_compressed

def DataLoader(face):
    
    n_train = 2
    n_test = 1
    # split the data into to train and test set
    # pick up two random image of each subject and appened it to the train set
    # append remaining image of each subject to test set
    for i in range(0, face.shape[1],3):
        
        if i == 0:
        
            each_class = face[:,i:i+3]
            each_class = each_class[:,np.random.permutation(each_class.shape[1])]
            x_train = each_class[:,:2]
            y_train = [0,0]
            x_test = each_class[:,2].reshape((each_class.shape[0],1))
            y_test = [0]
            continue
        
        each_class = face[:,i:i+3]
        # each_class = each_class[:,np.random.permutation(each_class.shape[1])]
        x_train = np.concatenate((x_train,each_class[:,:2]),axis=1)
        x_test = np.concatenate((x_test,each_class[:,2].reshape(each_class.shape[0],1)),axis = 1)
        for j in range(n_train):
            y_train.append(int(i/3))
        y_test.append(int(i/3))        

    return x_train, y_train, x_test, y_test
    
def main():
    data = sc.loadmat('data.mat')
    face = data ['face']
    face = face.reshape((-1,face.shape[2]))
    # compressed_face = PCA_compressed(face)
    # _,_,_,_ = DataLoader(compressed_face)
    
    
if __name__ == '__main__':
    main()