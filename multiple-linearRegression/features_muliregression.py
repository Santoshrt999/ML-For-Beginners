import numpy as np
import matplotlib.pyplot as plt
import math, copy

X_train=np.array([[952,2,1,65],[1244,3,2,64],[1947,3,2,17]])
Y_train=np.array([271.5,232,509.8])

X_features = ['size(sqft)','bedrooms','floors','age']

#plot graph for each feature here using price in dollats
# make 1 row of 4 subplots (one per feature); sharey=True so all panels share the same y-axis (price)
fig,ax=plt.subplots(1,4, figsize=(12,3), sharey=True)
for i in range(len(ax)):
    # X_train[:,i] = column i = feature i across all houses (x-axis) vs price (y-axis)
    ax[i].scatter(X_train[:,i], Y_train)
    ax[i].set_xlabel(X_features[i])      # label x-axis with this feature's name
ax[0].set_ylabel("price in dollars")     # only the leftmost panel needs the shared y-label

plt.show()                               # render the figure window


w_int=np.zeros(4)
b_int=0

#when computing gradient descent compute both 
def compute_gradient(x,y,w,b):
    m=x.shape[0]
    sum_dw=0
    sum_db=0
    for i in range(m):
        f_wb=np.dot(x[i],w)+b-y[i]
        sum_dw += f_wb*x[i]
        sum_db += f_wb
    sum_dw = sum_dw/m
    sum_db= sum_db/m    
    return sum_dw, sum_db



def gradient_descent(X,Y,w,b,alpha,iters):

    J_History=[]
    w=copy.deepcopy(w)
    b=copy.deepcopy(b)

    for i in range(iters):
        dj_dw,dj_db=compute_gradient(X,Y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
    
    return w,b


w_final, b_final = gradient_descent(X_train,Y_train,w_int,b_int,5.0e-7,1000)

print(f"W is {w_final} and B is {b_final}")
m,_=X_train.shape
for i in range(m):
    print(f"Predictions Value {np.dot(w_final,X_train[i])+ b_final:0.0f} target Value {Y_train[i]:0.0f} ")


# z-score normalization
# After z-score normalization, all features will have a mean of 0 and a standard deviation of 1.

# To implement z-score normalization, adjust your input values as shown in this formula:
# 𝑥(𝑖)𝑗=𝑥(𝑖)𝑗−𝜇𝑗𝜎𝑗(4)
# where  𝑗
#   selects a feature or a column in the  𝐗
#   matrix.  µ𝑗
#   is the mean of all the values for feature (j) and  𝜎𝑗
#   is the standard deviation of feature (j).
# 𝜇𝑗𝜎2𝑗=1𝑚∑𝑖=0𝑚−1𝑥(𝑖)𝑗=1𝑚∑𝑖=0𝑚−1(𝑥(𝑖)𝑗−𝜇𝑗)2
def zscore_normalize_features(X):


    mu=np.mean(X, axis=0)
    sigma=np.std(X,axis=0)

    X_Norm = (X-mu) / sigma

    return (X_Norm, mu, sigma)


mu     = np.mean(X_train,axis=0)   
sigma  = np.std(X_train,axis=0) 
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma      

# 3 panels showing the SAME feature pair (size vs age) at three normalization stages
fig,ax=plt.subplots(1, 3, figsize=(12, 3))

# Panel 0: raw data - column 0 (size) on x-axis vs column 3 (age) on y-axis
ax[0].scatter(X_train[:,0], X_train[:,3])
ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[0].set_title("unnormalized")
ax[0].axis('equal')                          # equal x/y scale so the scale difference is visible

# Panel 1: mean-centered data (X - mu) - cloud shifts toward the origin
ax[1].scatter(X_mean[:,0], X_mean[:,3])
ax[1].set_xlabel(X_features[0]); ax[1].set_ylabel(X_features[3]);
ax[1].set_title(r"X - $\mu$")                # $\mu$ renders the Greek letter via matplotlib mathtext
ax[1].axis('equal')

# Panel 2: fully z-score normalized (X - mu)/sigma - cloud becomes a compact, round blob
ax[2].scatter(X_norm[:,0], X_norm[:,3])
ax[2].set_xlabel(X_features[0]); ax[2].set_ylabel(X_features[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')

fig.suptitle("distribution of features before, during, after normalization")  # one title over all 3 panels
plt.show()


# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")



#run gradient descent Now optional implement cost function from other class
w_norm, b_norm = gradient_descent(X_norm, Y_train,w_int,b_int, 1.0e-1, 1000)

print(f"w_norm {w_norm} and b_norm is {b_norm}")

m = X_norm.shape[0]
yp = np.zeros_like(Y_train)
for i in range(m):
    #Predict formula fx=wx+b
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features    
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],Y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp,color='orange', label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()


# Predict the price of a brand-new house [size, bedrooms, floors, age]
x_house = np.array([1200, 3, 1, 40])
# Must scale the new input with the SAME mu/sigma learned from training data
x_house_norm = (x_house - X_mu) / X_sigma
print(f"normal price is {x_house_norm}")
# Apply the trained model f(x) = w.x + b on the normalized input
x_house_predict=np.dot(x_house_norm, w_norm) + b_norm
# Targets were in thousands of dollars, so multiply by 1000 to get the real price
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")

