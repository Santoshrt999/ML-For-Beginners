import numpy as np
import matplotlib.pyplot as plt
import copy, math

from features_muliregression import compute_gradient, gradient_descent, zscore_normalize_features

# ---------- SECTION 1: NO feature engineering (raw x as the only feature) ----------
# Target is a parabola y = 1 + x^2, but we feed the model only x (a straight-line feature).
x = np.arange(0, 20, 1)          # input values 0..19
y = 1 + x**2                     # TARGET we want to predict (a curve)
X = x.reshape(-1, 1)             # (20,1): 20 examples x 1 feature (x)

print(f"x array is {x}")
print(f"X is {X}")
print(X.shape)

w_init = np.zeros(X.shape[1])   # one weight per feature (here 1) - must be an ARRAY, not scalar 0.
# x is small (0..19) so a moderate alpha=1e-2 is safe here
model_w,model_b = gradient_descent(X,y,w_init,0.,alpha = 1e-2,iters=1000)

# plt.scatter(x, y, marker='x', c='r', label="Actual Value");   # red x = actual data points
# plt.title("no feature engineering")
# # a LINEAR model can only draw a straight line -> it can't bend to fit the parabola
# plt.plot(x,np.dot(X,model_w)+model_b, label="Predicted Value");
# plt.xlabel("X");
# plt.ylabel("y");
# plt.legend();
#plt.show()


#quadratic equation
x = np.arange(0, 20, 1) #
y = 1 + x**2
# Engineer features 
X = x**2      #<-- added engineered feature
X=X.reshape(-1,1)
print(X.shape)

# model_w,model_b=gradient_descent(X,y,w_init,0.,alpha=1e-5, iters=10000)
# plt.scatter(x,y,marker='x',c='r',label='actual value')
# plt.title("x^2 feature")
# plt.plot(x, X@model_w+model_b, label="Predicted Value")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
#plt.show()

#cubic equation
# X=x**3 
# X=X.reshape(-1,1)

# model_w,model_b=gradient_descent(X,y,w_init,0.,alpha=1e-7, iters=10000)
# plt.scatter(x,y, marker='x', c='r', label="Actual")
# plt.title("x^3 feature")
# plt.plot(x, np.dot(X,model_w)+model_b, label='Predicted Value')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()


# create target data
x = np.arange(0, 20, 1)
y = x**2
# engineer features .
X = np.c_[x, x**2, x**3] #np.column stack

print(f"X shape is {X.shape}")
print(f"X1 shape is {X.shape[1]}")

# w_init=np.zeros(X.shape[1])   # one weight per FEATURE (3 features: x, x^2, x^3)


# model_w,model_b = gradient_descent(X, y,w_init, 0. ,alpha=1e-7, iters=100000)

# plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
# plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature
X_features = ['x','x^2','x^3']

# fig,ax=plt.subplots(1,3, figsize=(12,3), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X[:,i],y)#Get all rows and ith column and plot it with y i.e, x**2
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("y")
# plt.show()


#adding z-score to data
# create target data
x = np.arange(0,20,1)
X = np.c_[x, x**2, x**3]
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization 
X_norm, X_mu, X_sigma = zscore_normalize_features(X)     
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

print(f"X shape is {X.shape[1]}")
w_init=np.zeros(X.shape[1])

# train on X_norm (the normalized data) - NOT raw X - so the big alpha is safe
# model_w,model_b = gradient_descent(X_norm, y,w_init, 0. ,alpha=1e-3, iters=10000)

# plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
# # predict with X_norm too (model was trained on normalized features)
# plt.plot(x, X_norm@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()



#Complex Data Prediction
x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X_norm, X_mu, X_sigma = zscore_normalize_features(X) 

print(X.shape)
print(f"X1 SHAPE {X.shape[1]}")

w_init=np.zeros(X.shape[1])


model_w,model_b = gradient_descent(X_norm, y, w_init, 0., alpha = 1e-1, iters=199999)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X_norm@model_w + model_b, label="Predicted Value"); plt.xlabel("x Norm"); plt.ylabel("y"); 
plt.legend();
plt.show()