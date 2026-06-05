"""
@author: Santosh Ravi Teja Goteti
"""
#Gradient Descent Algorithm

# The whole loop in one picture

# repeat many times:
#     1. measure the slope (derivative)
#     2. step downhill by  α × slope
#     3. update w and b
# until the cost stops dropping

import matplotlib.pyplot as plt
import numpy as np
import math, copy

x_train = np.array([1.0, 2.0])   # sizes in THOUSANDS of sqft
y_train = np.array([300.0, 500.0]) # prices in THOUSANDS of dollars

# Step 1: Compute the cost J(w,b) for linear regression.
def compute_cost(x,y,w,b):
    m=x.shape[0]
    sum_cost=0
    for i in range(m):
        f_x= w*x[i]+b-y[i]   #prediction - actual
        cost=f_x**2
        sum_cost+=cost
    return (1/2*m) * sum_cost

#Step 2: Compute the gradient dw and db
def compute_gradient(x,y,w,b):
    m=x.shape[0]
    sum_dw=0
    sum_db=0
    for i in range(m):
        f_x=w*x[i]+b-y[i]   #prediction - actual common for dw and db
        sum_dw += f_x*x[i] #dw = (1/m) * sum of (prediction - actual) * x
        sum_db += f_x       #db = (1/m) * sum of (prediction - actual)
    dw=(1/m) * sum_dw
    db=(1/m) * sum_db
    return dw,db

#Step 3: Update w and b
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost, step 1 above
      gradient_function: function to call to produce gradient step, step 2 above
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] """
    

    J_history = []    
    p_history = []
    w = copy.deepcopy(w_in) #avoid modifying global w_in
    b = copy.deepcopy(b_in) #avoid modifying global b_in

    for i in range(num_iters):
        dw, db = gradient_function(x,y,w,b)   #Step 2: Compute the gradient dw and db
        w= w-alpha *dw #Step 3: Update w and b
        b= b-alpha *db #Step 3: Update w and b
        if i<100000: # prevent resource exhaustion
            J_history.append(cost_function(x,y,w,b)) #Step 1: Compute the cost J(w,b) for linear regression.
            p_history.append([w,b]) #append the parameters w,b at each iteration to p_history

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dw:0.3e}, dj_db: {db:0.3e}  ",
                  f"w: {w:0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history

# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")



fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()

print(f"1000 sqft house prediction ${w_final*1.0 + b_final:0.0f}K")
print(f"1200 sqft house prediction ${w_final*1.2 + b_final:0.0f}K")
print(f"2000 sqft house prediction ${w_final*2.0 + b_final:0.0f}K")
