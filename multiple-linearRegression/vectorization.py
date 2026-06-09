import numpy as np
import time
import math, copy

# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,));             print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

np.arange(4) # array of 0,1,2,3
np.arange(0, 4) # array of 0,1,2,3

np.arange(5)        # → [0, 1, 2, 3, 4]      stop at 5 (not included)
np.arange(2, 7)     # → [2, 3, 4, 5, 6]      start=2, stop=7
np.arange(0, 10, 2) # → [0, 2, 4, 6, 8]      start, stop, step

#vector slicing operations
a = np.arange(10)
print(f"a         = {a}")

#access 5 consecutive elements (start:stop:step)
c = a[2:7:1];     print("a[2:7:1] = ", c)

# access 3 elements separated by two 
c = a[2:7:2];     print("a[2:7:2] = ", c)

# access all elements index 3 and above
c = a[3:];        print("a[3:]    = ", c)

# access all elements below index 3
c = a[:3];        print("a[:3]    = ", c)

# access all elements
c = a[:];         print("a[:]     = ", c)



# NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")


a=np.arange(10)

print(f"Before: a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
print(f"a shape is {a[2].shape}")

try:
    c=a[10]
except IndexError as e:
    print(f"Error: {e}")


a = np.array([1,2,3,4])
print(f"a             : {a}")
# negate elements of a
b = -a 
print(f"b = -a        : {b}")

# sum all elements of a, returns a scalar
b = np.sum(a) 
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2      : {b}")


a = np.array([ 1, 2, 6, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

a = np.array([1, 2, 3, 4])
# multiply a by a scalar
b = 5 * a 
print(f"b = 5 * a : {b}") # multiply a by a scalar
#[5, 10, 15, 20]

#DOT FUNCTION is a*b = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[n-1]*b[n-1] where n is the number of elements in a and b. a and b must have the same number of elements. The result is a scalar.

def dot(a,b):
    sum=0;
    for i in range(len(a)):
        sum+=a[i]*b[i]
        print(f"sum = {sum}")
    return sum

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
print(f"dot(a,b) = {dot(a,b)}")

print(f"np.dot(a,b) = {np.dot(a,b)}")



X = np.array([[1,2],[2,3],[3,4],[4,5]])#2D array with shape (4,2) not (4,1)
Y=np.array([5,6,7,8])#1D array with shape (4,) not (4,1)
w = np.array([2,3])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")#2D array has shape (1,2) not (1,)
print(f"Y has shape {Y.shape}")#1-D array has shape (1,) not (1,1)
print(f"w has shape {w.shape}")#1 D ARRAY
print(f"c has shape {c.shape}")

print(f"c = np.dot(X[1], w) = {c}")

a = np.zeros((1, 5))                                       
print(f"a shape = {a.shape}, a = {a}")                     
a = np.zeros((2, 1))                                                                   
print(f"a shape = {a.shape}, a = {a}") 
a = np.random.random_sample((1, 1))  
print(f"a shape = {a.shape}, a = {a}")


a = np.arange(6).reshape(-1, 2) #-1 is a placeholder that tells NumPy to automatically determine the appropriate number of rows based on the total number of elements and the specified number of columns (2 in this case). The resulting array will have 3 rows and 2 columns, since 6 elements can be arranged into 3 rows of 2 columns each.
print(f"a.shape: {a.shape}, \na= {a}")

#access an element
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

#access a row
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")

a=np.arange(20).reshape(2,10)
print(f"a.shape: {a.shape}, \na= {a}")
print(f"a[0].shape is {a[0].shape}")
#a.shape is     (2, 10), and a[0],shape is (10,)

print(f"a[0,2:7:1] is {a[0,2:7:1]}")

#access 5 consecutive elements (start:stop:step) in both rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

print(f"a[:] is {a[:]} and {a[:].shape} is the same shape as a")

print(f"all elemennts in one row a[0] is {a[0]}")
print(f"all elemennts in one row a[1] is {a[1:]} and shape is {a[1].shape}")









