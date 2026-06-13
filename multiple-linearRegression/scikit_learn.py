import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=2)


X_train=np.array([[952,2,1,65],[1244,3,2,64],[1947,3,2,17]])#2D Array Features
Y_train=np.array([271.5,232,509.8])

X_features = ['size(sqft)','bedrooms','floors','age']


scaler = StandardScaler() #z score normalisation
X_norm = scaler.fit_transform(X_train) #z score using scalar.fit_transform
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")#max-min   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(X_norm,Y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

w_norm=sgdr.coef_
b_norm=sgdr.intercept_

print(f"Weights are w {w_norm} and base is {b_norm}")

y_predict=sgdr.predict(X_norm)

y_manual_predict=np.dot(X_norm,w_norm)+b_norm

print(f"Library predict {y_predict} and numpy predict {y_manual_predict}")




fig,ax=plt.subplots(1,4, figsize=(12,3), sharey=True)#use 1 row and 4 features for graph modelling
for i in range(len(ax)):
    ax[i].scatter(X_norm[:,i],Y_train, label="target")
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_norm[:,i], y_predict, label="prediction")
ax[0].set_ylabel("Price in Dollars")
ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")#for subplots use this
plt.show()




