import numpy as np 
import pandas as pd 
from model_trainer import ModelTrainer 
x3 = x2 = x1 = np.arange(-50,50,0.1)

epsilon = np.random.normal(0,1.0,size=len(x1))

#Some funny function to approximate 
#y = 10*x3*np.sin(x3) + x3**2 + x3 + 2*x2**3 + 23*x2 + np.cos(x1) + x1 + 69 + epsilon 
y = 3 * np.sin(x1) + 0.5*x2**3 - 2*x2**2 + np.cos(2*x3) + np.exp(-0.5*x3**2 )+4*x1*x2 + epsilon

X = pd.DataFrame({"x1":x1,"x2":x2,"x3":x3})


model =  ModelTrainer(X,y)
model.train_model()
(BETA,best_lambda) = model.train_lambdas() 
#best_lambda = np.float64(0.0013894954943731374)
#BETA = model.BETA(best_lambda)
X_raw = X.to_numpy()
y_hat = np.array([model.predict(xi,BETA) for xi in X_raw])
ss_res = np.sum((y-y_hat)**2) 
ss_tot = np.sum((y-np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"R^2 : {r2}")
print(f"lambda : {best_lambda}")
print(f"BETA:",BETA.flatten())
print(f"Number of parameters : {len(BETA)}")