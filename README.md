# Multiclass_Classfication-Octave
# Univariate And Multivariate Linear Regression-Octave
This project is part of the **Machine Learning** course offered by **Andrew Ng**.

# Project documantation
The project has following classes and script files.

# IrCostFunction.m -- *Script File*

**IrCostFunction.m** file has `lrCostFunction()` method, that is used to find cost **J** and **gradient** for multivariate logistic regression.
The return values of this method are **J**  and **gradient** `(n+1) x 1`. **J_history** keeps **J** value in each iteration. The method looks like in octave `function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)`, 
where **X** is `m x (n+1)` matrix, **y** is `m x 1` matrix, **alpha** is learning rate, and num_iters defines how many times to iterate.

# gradientDescentMulti.m -- *Script File*
**gradietDescentMulti.m** file has `gradientDescentMulti()` method, that is used to find optimized **theta** for univariate linear regression.
The return values of this method are **J_history**  and optimized **theta**. **J_history** keeps **J** value in each iteration. The method looks like in octave `function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)`, 
where **X** is `m x (n+1)` matrix, **y** is `m x 1` matrix, **alpha** is learning rate, and num_iters defines how many times to iterate. **gradietDescentmulti.m** file has the same structure as  **gradietDescent.m** file. Both has vectorized implementation of gradient descent. vectorized implementation doesn't use loop for matrix related computation.

# computeCost.m -- *Script File*
**computeCost.m** file has `computeCost()` method, that is used to compute cost **J** for univariate linear regression. The return value of this method is **J**. The method looks like in octave `function J = computeCost(X, y, theta)`, 
where **X** is `m x (n+1)` matrix, **y** is `m x 1` matrix, **theta** is `(n+1) x 1` matrix. It uses vectorize implementation of cost **J**.

# computeCostMulti.m -- *Script File*
**computeCostMulti.m** file has `computeCostMulti()` method, that is used to compute cost **J** for univariate linear regression. The return value of this method is **J**. The method looks like in octave `function J = computeCost(X, y, theta)`, 
where **X** is `m x (n+1)` matrix, **y** is `m x 1` matrix, **theta** is `(n+1) x 1` matrix. It uses vectorize implementation of cost **J**.
# normalEqn.m -- *Script File*
**normalEqn.m** file has `normalEqn(X,y)` method, which computes the **theta** using matrix computation, instead of gradient descent algorithm. The method looks like in octave `function [theta] = normalEqn(X, y)`, where **X** is `m x (n+1)` matrix and **y** is `m x 1` matrix.   
**NOTE:** In the normal equation computation, `m` should be greater than `n` because **X** is non-invertible if `m < n`, and may be non-invertible if `m = n`. We can solve non-invertable issue by using regularization term. See **Regularized Linear Regression** video of **Machine Learning** course by **Andrew Ng**.
# plotData.m -- *Script File*
**plotData.m** file has plotData(X,y) method, that is used to visualize **X** `(m x 1)`, **y** `(m x 1)`, and univariate regression line. 
# ex1.m -- *Script File*
This is the file that should be run for univariate linear regression.This file loads ex1data1.txt, get univariate linear regression model, predict price and so on.
# ex1_multi.m -- *Script File*
This is the file that should be run for multivariate linear regression.This file loads ex2data2.txt, get multivariate linear regression model, predict price and so on.

# ex1.pdf 
This PDF file has the detail description of the homework.

# NOTE 
Cost **J** and **gradient** of linear regression can be regularized to prevent overfitting and underfitting problem. Regularized cost **J** and **gradient** can be found on course lecture of week 3.



