"""
@author: Santosh Ravi Teja Goteti
"""
import numpy as np

# ======================================================================
# NOTES: Squared Error Cost Function
# ======================================================================
#
# The model (linear regression):   f_wb(x) = w*x + b
#
# The COST FUNCTION measures how wrong the model is. It compares each
# prediction f_wb(x^(i)) with the actual value y^(i):
#
#                  1     m-1
#    J(w,b)  =  ------  SUM  ( f_wb(x^(i)) - y^(i) )^2
#                 2m    i=0
#
# Reading the formula piece by piece:
#   f_wb(x^(i)) - y^(i)   -> the ERROR for one example (prediction - actual)
#   ( ... )^2             -> SQUARE it, so:
#                              * negative & positive errors don't cancel
#                              * large errors are punished much more
#   SUM (i = 0 .. m-1)    -> add up the squared errors of ALL examples
#   1/m                   -> take the AVERAGE (so dataset size doesn't matter)
#   1/2                   -> a convention; makes the calculus cleaner later
#                            when we differentiate for gradient descent
#
# Key intuition:
#   - SMALLER J  = BETTER fit.
#   - J = 0      = the line passes through every point perfectly.
#   - Goal of training = find the w, b that MINIMISE J.
#
# ======================================================================

x_train = np.array([1.0, 2.0])   # house size  (input  x)
y_train = np.array([100, 300])   # house price (target y)

#Homework: implement the cost function compute_cost() and test it with w=200, b=-100 and w=50, b=100

def compute_cost(x, y, w, b):
    """
    Computes the cost J(w,b) for linear regression.

    Args:
        x (ndarray): Array of shape (m,) containing the input data (house sizes).
        y (ndarray): Array of shape (m,) containing the target values (house prices).
        w (float): Weight parameter of the linear model.
        b (float): Bias parameter of the linear model."""
    m = x.shape[0]
    sum_cost = 0
    for i in range(m):
        f_x=w*x[i]+b - y[i] #prediction - actual
        cost = f_x ** 2 #squared error
        sum_cost += cost #sum of squared errors
    return 1/(2*m) * sum_cost #J(w,b) = 1/(2m) * sum of squared errors

# Quick check:
#   w=200, b=-100 fits this data perfectly  -> cost should be 0
#   w=50,  b=100  is a poor guess           -> cost should be large
print(f"Cost (w=200, b=-100): {compute_cost(x_train, y_train, 200, -100)}")
print(f"Cost (w=50,  b=100 ): {compute_cost(x_train, y_train, 50, 100)}")
