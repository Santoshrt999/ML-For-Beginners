import numpy as np
import math, copy
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

input_array = np.array([1,2,3])
exp_array=np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

#e is the Euler's number with value 2.71828

def sigmoid(z):

    g=1/(1+np.exp(-z))

    return g


#Arrange values between -10 to 10
z_tmp = np.arange(-10,11)
print(f"z_temp is {z_tmp}")

y=sigmoid(z_tmp)

np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

#plot z vs sigmoid(z)

fig,ax=plt.subplots(1,1, figsize=(12,3))
ax.plot(z_tmp, y)
ax.set_title('plot z vs sigmoid(z)')
ax.set_xlabel("z value")
ax.set_ylabel("sigmoid(z)")
plt.legend()
plt.show()

#calculating z using linear regression

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_init=np.zeros([1])
b=0.

#compute cost multiple regression

#Use this dot product formula to calculate y for m no.of features/training examples
def calculateY(x,w,b):
    m=len(x)
    z=np.zeros(m)
    for i in range(m):
        z[i]=w*x[i] + b
    return z

def calculateY_dot(x,w,b):
    return np.dot(x,w) + b


y_cal=calculateY(x_train,w_init,b)
#call sigmoid

g=sigmoid(y_cal)

print(f"sigmoid g is {g}")

X=x_train.reshape(-1,1)
print(f"X is {X.shape}")
print(f"X[0] is {X.shape[0]}")
print(f"X[1] is {X.shape[1]}")
w_init=np.zeros(X.shape[1])

y_cal2=calculateY_dot(X,w_init,b)

g=sigmoid(y_cal2)

print(f"sigmoid g for ycal2 is {g}")










