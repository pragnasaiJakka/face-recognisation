[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Project Description

The goal of this project is to implement different classifiers to achieve face recognition. You are given a set of faces and their corresponding labels. The given data set is divided into training and a testing set is used to train the classifiers.

There are two tasks:
1. Identifying the subject label from a test image.
2. Neutral vs. facial expression classification

# Pipeline

In this project, the following classifiers are implemented for face recognition.

**1. Bayes' Classifier:** Assuming the underlying distribution is Gaussian, you need to implement the
Maximum Likelihood estimation with Gaussian assumption followed by Bayes’
classification.

**2. k-NN Rule:** Implement the k-Nearest Neighbors (k-NN) rule to classify the test data. Vary
k to see its effect.

**3. Kernel SVM:** Implement the Kernel SVM classifier by solving the dual optimization problem. Use the Radial Basis Function (RBF) and the polynomial kernel. Choose the optimal values of &sigma;<sup>2</sup> and r using cross-validation.

**4. Boosted SVM:** Implement the AdaBoost algorithm for the class of linear SVMs. Investigate the improvement of the boosted classifiers with respect to the iterations of
AdaBoost, and compare it with the performance of the Kernel SVMs. 

**5. PCA:** Implement Principal Component Analysis (PCA) and apply it to
the data before feeding it into the classifiers above.

**6. MDA:** Similar to PCA, implement MDA followed by the training/appli-
cation of the classifiers.

# Dataset
Dataset consists of 600 images of 200 subjects. Each subject has one neutral face image and one image with expression and the last image is of neutral face but varied illumination.

# Results and Analysis
## Face Recognition

**1. PCA Analysis:**

![1](https://user-images.githubusercontent.com/90370308/217112368-6b8d18d5-b613-46de-b71e-ff3a67ba989e.png)![PCA image](https://user-images.githubusercontent.com/90370308/217111283-4f7fbd50-98e7-4e8e-9448-d34221691c23.png)

The above figure shows the graph of cumulative data variance vs number of eigen values. As shown in the graph, we can say that about 90% of the data variance are containing in top 100 eigen values (out of 504 eigen values). After 100 eigen values, the rate of change of variance with respect to number of eigen value is very small. Hence, to go from 90% to 95%, we need to add 100 more eigen value which is computationally expensive. Hence, in PCA compression, top 100 eigen vector are taken. So, the data is transformed from 504 dimension to 100 dimension.

**2. MDA Analysis:**

![2](https://user-images.githubusercontent.com/90370308/217113155-4b1a99c9-b94f-4c95-ad91-a4cd22e68cb7.png)![MDA image](https://user-images.githubusercontent.com/90370308/217113192-455497dd-9f88-4c46-b3de-be5a9d33988b.png)

The above figure shows the graph of cumulative data variance vs number of eigen values. As shown in the graph, we can say that about 95% of the data variance are containing in top 100 eigen values (out of 504 eigen values). After 100 eigen values, the rate of change of variance with respect to number of eigen value is very small. Hence, to go from 95% to 98%, we need to add 100 more eigen value which is computationally expensive. Hence, in MDA compression, top 100 eigen vector are taken. So, the data is transformed from 504 dimension to 100 dimension.

**3. Bayes' Classifier:**
- Optimal Parameter : Top 80 Eigen Value for PCA and MDA
- Test Accuracy:
  1. PCA : 57.0%
  2. MDA : 53.0% 

![3](https://user-images.githubusercontent.com/90370308/217146754-1800b7a9-4dd0-4e06-b324-86f785b81495.png)![4](https://user-images.githubusercontent.com/90370308/217146948-77b360fd-8b1c-4449-9b14-ea9ab44a4b4b.png)

**4. K-NN Rule:**
- Optimal Parameter : k = 2
- Test Accuracy:
  1. PCA : 54.0%
  2. MDA : 52.5%
  
![face_knn_PCA](https://user-images.githubusercontent.com/90370308/217147991-858e565a-41c8-4e3c-8b3f-6220665b6b6d.png)![face_knn_MDA](https://user-images.githubusercontent.com/90370308/217148008-93a359a7-16f9-4905-bf5a-cdad25db8f59.png)

As we can see in the above graph of k-NN with PCA and MDA, the highest
accuracy is achieved for k = 2 (which is equal to the number of data of each class in
the train set). After that further increase in k has little impact on the accuracy and it
decreases once it go beyond 6.
Also, note that in this experiment if the value of k is even and if there are equal number of neighbor of
two classes, the classification will be done considering 1-NN rule.

## Neutral vs Expression Classification

**1. PCA Analysis:**

![PCA_NvsF](https://user-images.githubusercontent.com/90370308/217148684-471258c0-0589-4fa5-a0ee-a9d72a02f1b6.png)

The above figure shows the graph of cumulative data variance vs number of eigen values. As shown in
the graph, we can say that about 92% of the data variance are containing in top 100 eigen values (out of
504 eigen values). After 100 eigen values, the rate of change of variance with respect to number of
eigen value is very small. Hence, to go from 92% to 95%, we need to add 100 more eigen value which
is computationally expensive. Hence, in PCA compression, top 100 eigen vector are taken. So, the data
is transformed from 504 dimension to 100 dimension.

**2. MDA analysis:**

![MDA_NvsF](https://user-images.githubusercontent.com/90370308/217148803-022a2c75-9057-4f3f-b9cc-fa9aae72620b.png)

The above figure shows the graph of cumulative data variance vs number of eigen values. As shown in
the graph, we can say that about 100% of the data variance are containing in top 1 eigen values (out of
504 eigen values). After 1 st eigen values, the rate of change of variance with respect to number of eigen
value almost 0. This is very surprising. This might be due to the face that since we have 360 data points
for only two classes, MDA might found the best 1 direction where almost all the images are  100 % separable.

**3. Bayes' Classifier:**
- Optimal Parameters 
  1. Top 60 eigen value in PCA analysis
  2. Top 60 eigen value in MDA analysis
  3. 90% data in training and 10% in testing
- Test accuracy 
  1. PCA – 92.5% 
  2. MDA – 77.5%

![PCA_NvsF_Bayes_lam](https://user-images.githubusercontent.com/90370308/217149660-874408bd-0608-49d7-b0c3-6a29f181de4b.png)
![MDA_NvsF_Bayes_lam](https://user-images.githubusercontent.com/90370308/217149677-9474fdb9-466c-43b0-bb87-f88e62631a6c.png)
![PCA_NvsF_Bayes_train](https://user-images.githubusercontent.com/90370308/217149737-a731a9d3-3c47-4f71-b6c5-40d9b063ce5c.png)
![MDA_NvsF_Bayes_train](https://user-images.githubusercontent.com/90370308/217149764-0ee0d216-b706-4b39-89eb-43a7b2fe3b19.png)

The first two graphs show how test accuracy change with number of eigen value in MDA and PCA
analysis. Since we have large number of data for both the classes, data can be compressed in relatively
low dimension and we can achieve maximum test accuracy with top 60% eigen values.
The third and forth graph shows how test accuracy change with splitting of training and testing data. It
can be said that test accuracy is almost independent of this ratio. However, if we have more data intraining, model can generalize well on the future data. Hence, we take 90% data into training and 10%
data in the testing.

**4. k-NN rule:**
- Optimal Parameters
  1. Top 60 eigen value in PCA analysis
  2. Top 60 eigen value in MDA analysis
  3. 90% data in training and 10% in testing
  4. K = 5
 - Test Accuracy
  1. PCA - 50.0%
  2. MDA - 47.5%

![11](https://user-images.githubusercontent.com/90370308/217150896-6ebeb413-632d-471e-a0b4-3ef2f19436ce.png)
![12](https://user-images.githubusercontent.com/90370308/217150923-8af26aba-70c5-40bc-bd70-da0b78914ae7.png)
![13](https://user-images.githubusercontent.com/90370308/217150933-01addd48-418a-4278-8190-d2ed0cbf335e.png)
![14](https://user-images.githubusercontent.com/90370308/217150943-b54e57ad-71c7-453c-b439-b3c6c96a2be5.png)

  
**5. Support Vector Machine with RBF Kernel:**

- Optimal Parameters
  1. Top 60 eigen value in PCA analysis
  2. Top 60 eigen value in MDA analysis
  3. 80% data in training and 20% in testing
  4. &sigma;<sup>2</sup> = 3

- Test Accuracy:
  1. PCA - 83.75%
  2. MDA - 77.50%

![15](https://user-images.githubusercontent.com/90370308/217151343-2a830d75-57f6-4487-8766-c318072798fe.png)
![16](https://user-images.githubusercontent.com/90370308/217151359-7c6032c1-60d2-42bf-84f2-a82282e00cec.png)
![17](https://user-images.githubusercontent.com/90370308/217151381-9f6f844b-dff6-4c1d-a970-67f82bbde563.png)
![18](https://user-images.githubusercontent.com/90370308/217151393-6e7d9298-e5b6-4863-b9a4-7f666abe1c42.png)

**5. Support Vector Machine with polynomial Kernel:**

- Optimal Parameters
  1. Top 60 eigen value in PCA analysis
  2. Top 60 eigen value in MDA analysis
  3. 80% data in training and 20% in testing
  4. r = 2

- Test Accuracy:
  1. PCA - 70.0%
  2. MDA - 71.25%

![19](https://user-images.githubusercontent.com/90370308/217152121-a083f4d0-38f7-4efd-902d-67749f2fdf7f.png)
![20](https://user-images.githubusercontent.com/90370308/217152131-1cc2138c-6896-495f-8ad0-45e85a31f1a1.png)
![21](https://user-images.githubusercontent.com/90370308/217152194-d768558b-30dd-4649-aa8e-600de83e4e78.png)
![22](https://user-images.githubusercontent.com/90370308/217152235-b24e558a-59f4-45f6-97d3-4909add8ede1.png)


**6. Boosted SVM:**

- Optimal Parameters
  1. Top 60 eigen value in PCA analysis
  2. Top 60 eigen value in MDA analysis
  3. 90% data in training and 10% in testing
  4. Number of Iteration : n = 8

- Test Accuracy:
  1. PCA - 94.5 %
  2. MDA - 93.0 %

![23](https://user-images.githubusercontent.com/90370308/217152486-e0d6eea6-bdbe-4cf0-b48c-aeaf3178b7ca.png)
![24](https://user-images.githubusercontent.com/90370308/217152496-d7d0e580-a2fb-43a7-83c3-e00b238d9a04.png)

# Requirement
Python 2.0 or above


















