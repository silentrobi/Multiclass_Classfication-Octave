# Multiclass Classfication-Octave
# Univariate And Multivariate Linear Regression-Octave
This project is part of the **Machine Learning** course offered by **Andrew Ng**.

# Project documantation
The project has following script files.

# IrCostFunction.m -- *Script File*

**IrCostFunction.m** file has `lrCostFunction()` method, that is used to find cost **J** and **gradient** for multivariate logistic regression. The return values of this method are **J**  and **gradient** `(n+1) x 1`. The method looks like in octave `function [J, grad] = lrCostFunction(theta, X, y, lambda)`, where **X** is `m x (n+1)` matrix, **y** is `m x 1` matrix, and **lambda** is regularization parameter.

# oneVsAll.m -- *Script File*
**oneVsAll.m** file has `oneVsAll()` method, that is used to find optimized **k** number of **thetas**, where **k** means the number of classes, and k >=3 for multiclass classification. The method uses advanced optimization algorithm `fmincg` to find the **thetas*. 
The return values of this method are **all thetas** `k x (n+1)` for all classes. The method looks like in octave `function [all_theta] = oneVsAll(X, y, num_labels, lambda)`, where **X** is `m x n` matrix, **y** is `m x 1` matrix, **num_labels** means total number of classes, and **lambda** is regularization parameter. 

# predictOneVsAll.m -- *Script File*
**predictOneVsAll.m** file has `predictOneVsAll()` method, that is used to find **p**, where **p** `(m x1)` is a vector of predicted classes for training data. The method looks like in octave `function p = predictOneVsAll(all_theta, X) `, where **X** is `m x n` matrix, **all_theta** is `k x (n+1)` matrix.

# sigmoid.m -- *Script File*
**sigmoid.m** file has `sigmoid()` method, that is used to compute sigmoid of input vector/matrix.

# predict.m -- *Script File*
*predict.m** file has `predict()` method, that is used to find **p**, where **p** `(m x1)` is a vector of predicted classes for training data. The method looks like in octave `function p = predict(Theta1, Theta2, X) `, where **X** is `m x n` matrix, **Theta1** is `25 x (n+1)`  and **Theta1** is `(k x 26)`.

# displayData.m -- *Script File*
**displayData.m** file is used to display image data from training set.
# ex3.m -- *Script File*
This file runs  multiclass logistic regression classifier. This file loads `ex3data1.mat`, gets multiclass logistic regression model, and predict the class of an input data. `ex3data1.mat` file has image data.
# ex3_nn.m -- *Script File*
This file runs nueral network of multiclass logistic regression classifier. **p** vector is calculated using neural network **forward propogation** algorithm. This file loads `ex3data1.mat` and `ex3weight.mat`, gets neural network of multiclass logistic regression model. `ex3weight.mat`has optimized **Thetas** that is found by **backward propogation algorithm**. This algorithm is part of homework 4. `ex3data1.mat` file has image data.

# ex3.pdf 
This PDF file has the detail description of the homework.

**n** is number of features, **m** is number of training examples, and **k** is number of classes.
