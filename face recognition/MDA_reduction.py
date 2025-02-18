import scipy.io as sc
import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader

def MDA_train(face):
    
    # class information
    n_class = 200.0
    n_data = 2
    total_data = 400.0
    #Probability of each class
    P = n_data / total_data
    
    #take the mean of all the images
    # mean will be flattened array of size (504,)
    
    mean_face = np.mean(face, axis = 1).reshape((face.shape[0],1))
    mean_class = np.zeros((face.shape[0],1))

    # mean of each class stack column wise
    for i in range(0,face.shape[1] - (n_data-1),n_data):
        if i==0:
            mean_class = np.mean(face[:,i:i+n_data],axis=1,keepdims=True)
            continue
        mean_class = np.concatenate((mean_class, np.mean(face[:,i:i+n_data],axis=1,keepdims=True)), axis=1)

    sigma_b = P * (mean_class - mean_face)@(mean_class - mean_face).T

    for i in range(face.shape[1]):
        
        each_class_mean = mean_class[:,int(i/n_data)].reshape((face.shape[0],1))
        
        if i==0:
            
            mean_centered_face = np.array(face[:,i].reshape((face.shape[0],1)) - each_class_mean)
            mean_centered_face_arr = mean_centered_face
            sigma_w = mean_centered_face@mean_centered_face.T
            continue
        
        mean_centered_face = np.array(face[:,i].reshape((face.shape[0],1)) - each_class_mean)
        mean_centered_face_arr = np.concatenate((mean_centered_face_arr, mean_centered_face),axis = 1)
        sigma_w += mean_centered_face@mean_centered_face.T
    
    sigma_w = (P*1/n_data)*sigma_w + (1.4*10e-4)*np.identity(sigma_w.shape[0])

    eig_val, eig_vect = np.linalg.eig(np.linalg.inv(sigma_w)@sigma_b)
    
    sort_index = np.argsort(eig_val)[::-1]
    
    sorted_eigen_value = eig_val[sort_index]
    
    sorted_eigen_vector = eig_vect[sort_index]
   
    direction = sorted_eigen_vector[:,0:80]
    
    # #Forward Projection
    # p_forward = direction.T@ mean_centered_face_arr
    
     #Forward Projection
    p_forward = direction.T@ face
    
    # #backward projection
    # compressed_face = (direction@p_forward).real
    
    # # add the mean after MDA compression
    # for i in range(compressed_face.shape[1]):

    #     if i == 0:
    #         each_class_mean = mean_class[:,int(i/n_data)].reshape((compressed_face.shape[0],1))
    #         result = compressed_face[:,i].reshape((compressed_face.shape[0],1)) + each_class_mean
    #         continue

    #     each_class_mean = mean_class[:,int(i/n_data)].reshape((compressed_face.shape[0],1))
    #     recorvered_face = compressed_face[:,i].reshape((compressed_face.shape[0],1)) + each_class_mean
    #     result = np.concatenate((result,recorvered_face),axis=1)
    
    return p_forward, mean_face, direction, sorted_eigen_value

def MDA_test(face, mean_face, direction):
    
    # mean_centered_face_arr = face - mean_face

    # #Forward Projection
    # p_forward = direction.T@ mean_centered_face_arr
    
    #Forward Projection
    p_forward = direction.T@ face
    
    #backward projection
    # compressed_face = (direction@p_forward).real
    
    # result = compressed_face + mean_face
    
    return p_forward
    
def main():
    # Read data.mat
    data = sc.loadmat('data.mat')
    face = data['face']
    #reshape data into matrix where each column is a flattened image
    face = face.reshape((-1,face.shape[2]))
    x_train, y_train, x_test, y_test = DataLoader(face)
    x_train, mean, eig_vect, p = MDA_train(x_train)
    x_test = MDA_test(x_test, mean, eig_vect)
    
    # Data variance plot
    p = p.real
    p = p*100/np.sum(p)
    for i in range(len(p)):
        if i > 0:
            p[i] = p[i-1] + p[i]
    
    plt.plot(p)
    plt.title('Cumulative Data Variance vs Eigen Value in MDA analysis')
    plt.xlabel('Number of eigen value (lambda)')
    plt.ylabel('Cumulative data variance (%)')
    plt.show()
      
    
if __name__ == '__main__':
    main()
    







