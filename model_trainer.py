import os
import numpy as np 
import pandas as pd 
from scipy.linalg import block_diag
# These are the different function we've set up
# They look ugly like that ,just look at the document to see what they really represent in a mathematical perspective
def h(x,xi):
    return (x-xi)**3 if x > xi else 0 
def h2(x,xi):
    return 6*(x-xi) if x > xi else 0
def omega_antiderivative(x,xk,xj):
    return 36 * ((1/3) * (x**3) - (1/2)*xk*(x**2) - (1/2)*xj*(x**2) + xj*xk*x )


class ModelTrainer:
    def __init__(self,X: pd.DataFrame,y: np.ndarray):
        self.X = X
        self.y=y
        self.transformations = [] # Matrix of lambda vectors
        self.build_transformations()
        self.B=[]
        self.OMEGA=[]
    def build_transformations(self):
        MAX_POLY=3
        n,p = self.X.shape 
        for j in range(p):
            xj = self.X.iloc[:,j]
            feature_transformations = []
            
            feature_transformations.append(lambda x: 1.0)
            feature_transformations.append(lambda x: x)
            feature_transformations.append(lambda x: x**2)
            feature_transformations.append(lambda x: x**3)

            for xi in xj:
               h_fn = lambda x,xi=xi: (x-xi)**3 if x > xi else 0
               feature_transformations.append(h_fn)
            self.transformations.append(feature_transformations)

    def b(self,x:np.ndarray)->np.ndarray:
        b = []
        for i in range(len(x)): # for each feature in vector x
            feature = x[i]
            feature_transformations = self.transformations[i]
            b_feature = []
            for transformation in feature_transformations: # for each lambda in transformations
                b_feature.append(transformation(feature))
            b.extend(b_feature)
        return np.array(b)
    def calc_B(self) -> np.ndarray:
        B_ks = []
        n,p = self.X.shape
        X = self.X.to_numpy()
        for j in range(p):
            xj = X[:,j]
            bias = np.ones((n,1))
            x_col = xj[:,None]
            x_col2 = xj[:,None]**2
            x_col3 = xj[:,None]**3


            B_k = np.array([[h(xi,xj_val) for xj_val in xj] for xi in xj])
            B_k = np.hstack([bias,x_col,x_col2,x_col3,B_k])
            B_ks.append(B_k)

        B = np.hstack(B_ks)
        return B
    def calc_OMEGA(self) -> np.ndarray:
        X = self.X.to_numpy()
        OMEGA_ks =[]
        n,p = X.shape
        for i in range(p):
            x = X[:,i]
            # I know it's hard to understand what it represent
            # In C++ I would have made a simple nested for loop where we calculate the integral f(t)*f(t) 
            # But doing this would be very unoptimised in Python, so we are vectorising those x's observations
            # in order for numpy to calculate all of them in it's C++ engine. 
            xj = x[:,None] 
            xk = x[None,:]

            a = np.maximum(xj,xk)
            b = np.max(x) 

            disabled_knots = a >= b 
            Fb = omega_antiderivative(b,xj,xk)
            Fa = omega_antiderivative(a,xj,xk)
            
            OMEGA_k = Fb - Fa 
            OMEGA_k[disabled_knots] = 0

            zeros_2x2 = np.zeros((4,4))
            zeros_2xn = np.zeros((4,n))
            zeros_nx2 = np.zeros((n,4))
            OMEGA_k = np.block([
                [zeros_2x2,zeros_2xn],
                [zeros_nx2,OMEGA_k]
            ])

            OMEGA_ks.append(OMEGA_k) 
        OMEGA = block_diag(*OMEGA_ks)
        return OMEGA
    def BETA(self,lambda_hparam) -> np.ndarray:
        n = self.X.shape[0]
        BETA = np.linalg.pinv( (self.B.T @ self.B) + (n*lambda_hparam * self.OMEGA)  ) @ self.B.T @ self.y
        return BETA 
    def J(self,lambda_hparam,BETA):
        n = self.X.shape[0]
        return (self.y - self.B@BETA ).T @ (self.y - self.B@BETA ) + n*lambda_hparam*(BETA.T@self.OMEGA@BETA)
    def predict(self,x :np.ndarray,BETA:np.ndarray):
        transformed_x = self.b(x)
        return transformed_x@BETA
    def save_model(self):
        os.makedirs("model",exist_ok=True)
        np.save("model/B.npy",self.B)
        np.save("model/OMEGA.npy",self.OMEGA)
    def load_model(self):
        self.B = np.load("model/B.npy",self.B)
        self.OMEGA = np.load("model/OMEGA.npy",self.OMEGA)
    def train_model(self):
        self.B = self.calc_B()
        self.OMEGA = self.calc_OMEGA()
    def S_lambda(self,lambda_hparam):
        n = self.X.shape[0]
        return self.B@ np.linalg.pinv(self.B.T@ self.B + n * lambda_hparam*self.OMEGA )@self.B.T
    def train_lambdas(self) -> tuple:
        lambdas = np.logspace(-5, 2, 50) 
        SCORE = []
        for i in range(len(lambdas)):
            x_lambda = lambdas[i]
            S_lambda = self.S_lambda(x_lambda)
            g_lambda = S_lambda @ self.y
            diag_S_lambda = np.diag(S_lambda)
            SCORE.append(np.sum( ((self.y - g_lambda) / (1-diag_S_lambda))**2 ))
        best_lambda = lambdas[np.argmin(SCORE)]
        best_BETA = self.BETA(best_lambda)

        return (best_BETA,best_lambda)
