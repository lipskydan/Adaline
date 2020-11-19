# Adaline
### Adaptive Linear Neuron by Python

This repository has two realisation of Adaline - **AdalineGD** and **AdalineSGD**.

The main difference is using different gradient descents. 
**AdalineGD** uses *Batch Gradient Descent* - weights are updated based on the sum of accumulated 
errors over all Xi samples.
**AdalineGD** uses *Stochastic Gradient Descent* - we update weights gradually for each training sample.

### Problem statement 
Determine the type of iris (setosa or versicolor) by the length of the sepal and the length of the petal.

### Dataset
Dataset is file 'iris.data'. Each realisation has this file in root.

### Visualisation

Each realisation has directory 'images' where we can see information about learning rate (simple one and after
standardisation. The meaning of word 'standardisation' will be explained in next subtitle) and result of binary 
classification.

### Standardisation

Standardization is one method for scaling features. This method gives the data the quality of a standard normal 
distribution, which facilitates faster convergence of gradient descent training.


