import scipy.io as sc
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader

def PCA_compressed_train(face):
    
    #take the mean of all the images
    # mean will be flattened array of size (504,)
    mean_face = np.mean(face, axis = 1)

    #subtract mean from each images
    shifted_face = face - np.reshape(mean_face,(-1,1))

    #covariance matrix of the data
    covarinace_matrix = (shifted_face@shifted_face.T)/shifted_face.shape[1]

    #find the eigen value and eigen vector of the covariance matrix
    eig_val, eig_vect = np.linalg.eig(covarinace_matrix)

    #sort eigen value and eigen vector
    sort_index = np.argsort(eig_val)[::-1]
    
    sorted_eig_val = eig_val[sort_index]

    sorted_eig_vect = eig_vect[:,sort_index]
    
    reduced_eig_vect = sorted_eig_vect[:, 0:80]
     
    #forward projection
    P = reduced_eig_vect.T@shifted_face

    #backward projection
    # reduced data in 100 dimension
    # reduced_face = reduced_eig_vect@P
    
    # shifted_reduced_face = (reduced_face + np.reshape(mean_face,(-1,1))).real
    
    return P, mean_face, reduced_eig_vect, sorted_eig_val

def PCA_compressed_test(face, mean_face, reduced_eig_vect):

    #subtract mean from each images
    shifted_face = face - np.reshape(mean_face,(-1,1))
     
    #forward projection
    P = reduced_eig_vect.T@shifted_face

    #backward projection
    # reduced data in 100 dimension
    # reduced_face = reduced_eig_vect@P
    
    # shifted_reduced_face = (reduced_face + np.reshape(mean_face,(-1,1))).real
    
    return P
    
    
def main():
    # Read data.mat
    data = sc.loadmat('data.mat')
    face = data['face']
    #reshape data into matrix where each column is a flattened image
    face = face.reshape((-1,face.shape[2]))
    x_train, y_train, x_test, y_test = DataLoader(face)
    x_train, mean, eig_vect, p = PCA_compressed_train(x_train)
    x_test = PCA_compressed_test(x_test, mean, eig_vect)
    
    # plot eigen value
    p = p.real
    p = p*100/np.sum(p)
    for i in range(len(p)):
        if i > 0:
            p[i] = p[i-1] + p[i]
    
    plt.plot(p)
    plt.title('Cumulative Data Variance vs Eigen Value in PCA analysis')
    plt.xlabel('Number of eigen value (lambda)')
    plt.ylabel('Cumulative data variance (%)')
    plt.show()

if __name__ == '__main__':
    main()
    







