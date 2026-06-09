import numpy as np
import matplotlib.pyplot as plt
import statistics
import math, copy

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])#Sqft, number of bedrooms, number of bathrooms, age of the house
y_train = np.array([460, 232, 178])#price in USD

print(f"x_train shape: {X_train.shape}")#(3,4)
print(X_train)
print(f"y_train shape: {y_train.shape}")#(3,)
print(y_train)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

#w is a vector of weights
#b is a scalar bias term

#𝑓𝐰,𝑏(𝐱)=𝑤0𝑥0+𝑤1𝑥1+...+𝑤𝑛−1𝑥𝑛−1+𝑏
#𝑓𝐰,𝑏(𝐱)=𝐰⋅𝐱+𝑏
#. is dot product



def predict_single(X,w,b):
    m=X.shape[0] #m is the number of training examples
    sum=0        # one predicted value per training exampl
    addB=0
    for i in range(m):
        sum+=w[i]*X[i]
    addB = sum + b   #adding b to the sum of w[i]*X[i] outside if loop
    return addB

print(f"Predicted value of y for the first training example: {predict_single(X_train[0], w_init, b_init)}")

def predict_builtIn(x,w,b):
    dotP=np.dot(x,w)
    return dotP+b    


f_wb = predict_builtIn(X_train[0], w_init, b_init)
print(f"f_wb prediction for single is {f_wb}")

print(f"shapes are {X_train.shape}, x_train[0] is {X_train[0].shape} and x_train[0].shape[0] is {X_train[0].shape[0]}")

#compute cost with multiple variables

#𝐽(𝐰,𝑏)=12𝑚∑𝑖=0𝑚−1(𝑓𝐰,𝑏(𝐱(𝑖))−𝑦(𝑖))2

def compute_costMultiple(X,Y,w,b):
      """ compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost"""
      m=X.shape[0]
      cost=0.0
      for i in range(m):
           f_wb= np.dot(X[i], w) + b #f(x)=w.x+b use np.dot for multiple variables cost function
           cost += (f_wb - Y[i])**2
      return (1 / (2 * m)) * cost
   

# Compute and display cost using our pre-chosen optimal parameters. 
cost = compute_costMultiple(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')


# 5 Gradient Descent With Multiple Variables¶
# Gradient descent for multiple variables:

# repeat} until convergence:{𝑤𝑗=𝑤𝑗−𝛼∂𝐽(𝐰,𝑏)∂𝑤𝑗𝑏  =𝑏−𝛼∂𝐽(𝐰,𝑏)∂𝑏for j = 0..n-1(5)

# where, n is the number of features, parameters 𝑤𝑗
# , 𝑏
# , are updated simultaneously and where

# ∂𝐽(𝐰,𝑏)∂𝑤𝑗∂𝐽(𝐰,𝑏)∂𝑏=1𝑚∑𝑖=0𝑚−1(𝑓𝐰,𝑏(𝐱(𝑖))−𝑦(𝑖))𝑥(𝑖)𝑗=1𝑚∑𝑖=0𝑚−1(𝑓𝐰,𝑏(𝐱(𝑖))−𝑦(𝑖))

def compute_gradient(x,y,w,b):
     m=x.shape[0]
     sum_dw=0
     sum_db=0
     for i in range(m):
          f_wb=np.dot(x[i],w)+b-y[i] #for mulitple variables always use np.dot function to multiple x-TRAIN first array vs w
          sum_dw+=f_wb*x[i]
          sum_db+=f_wb
     sum_dw =sum_dw/m
     sum_db=sum_db/m
     return sum_dw,sum_db

     #Compute and display gradient 
tmp_dj_dw, tmp_dj_db = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')



def gradient_descent(X, Y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    J_history = []    
    w = copy.deepcopy(w_in) #avoid modifying global w_in vector
    b = copy.deepcopy(b_in) #avoid modifying global b_in scalar.

    for i in range(num_iters):
         
         dj_dw,dj_db=gradient_function(X,Y,w,b)#call gradient here and record cost below in an array

         w=w-alpha*dj_dw
         b=b-alpha*dj_db

        #save cost at each iteration
         if i<100000:
              J_history.append(cost_function(X,Y,w,b))

        #PRINT cost at every 10 iterations
         if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing     


# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_costMultiple, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.0f}, target value: {y_train[i]}")


# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost v Iteration 1st 100 values"); ax2.set_title("Cost vs Iteration (tail)")
ax1.set_ylabel("cost");  ax2.set_ylabel("cost")
ax1.set_xlabel("iter step"); ax2.set_xlabel("iter step")
plt.show()
