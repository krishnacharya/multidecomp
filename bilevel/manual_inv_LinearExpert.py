import numpy as np
from bilevel.ExpertsAbstract import *
import pandas as pd

def sherman_inv(previnv, v):
    '''
        https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
        Sherman morrison Used to compute (A + uv^T)^{-1}
        same u = v = x
    '''
    nr = previnv @ v @ v.T @ previnv
    return previnv - nr / (1 + v.T @ previnv @ v)

class Manual_inv_LinearExpert(Expert):
  def __init__(self, X_dat_np: np.array, y_dat_np: np.array, l2_pen = 1.0):
    self.name = "Manual inversion"
    self.X_dat_np = X_dat_np
    self.y_dat_np = y_dat_np
    self.l2_pen = l2_pen
    # self.time = 0 # number of time slots this OLS expert has seen
    self.dim = X_dat_np.shape[1] # dimensionality of the actions

    # self.x_tarr = [] # x_t array, each x_t is in R^d, with L2 norm < 1
    # self.theta_predarr = np.zeros((self.dim, 1)) #theta_pred arr, stores the ridge regression \theta_t at each time t using t-1 history
    self.theta_pred = np.zeros(self.dim) # has shape (dim,)
    self.y_predarr = [] #y_hattarr, yhat t array, store the clipped [0, 1] values of the algorithm
    # self.y_tarr = [] # y_t array, labels are normalized to be [0,1]
    self.loss_tarr = [] # square loss arr for the algorithm, loss at time t (y_hatt- y_t)^2
    # self.cuml_loss_tarr = [0.0] #cumulative loss for algorithm with time, \sum_t (yhat-y_t)^2
    self.previnv = (1.0/ self.l2_pen) * np.identity(self.dim) # running A^{-1} required for ridge regression, initially set to identity, so always invertible sum of psd
    # self.previnv_for_noreglz = (1000) * np.identity(dim) # \lambda I inverse is 1/lambda I, here lambda i selected is 0.001, to be close to LS best hindsight
    # self.xt_labelprodsum = np.zeros((dim)) #sum of y_t x_t used in thetahat calculation (rhs of ridge regression)
    self.xt_labelprodsum = np.zeros(self.dim)
    # self.best_theta_hindsight = []
    # self.bestsqloss = []
    # self.best_theta_norm = []
    
  # def update_best_hindsightt(self):
  #   '''
  #     this is an internal method called by self.update_t(),
  #     calculates best theta in hindsight over t rounds, and corresponding square loss
  #   '''
  #   self.previnv_for_noreglz = sherman_inv(self.previnv_for_noreglz, self.x_tarr[-1])
  #   self.best_theta_hindsight.append(self.previnv_for_noreglz @ self.xt_labelprodsum) # best theta in hindsight Least squares solution, no Lambda I
  #   x_tmat = np.reshape(self.x_tarr, (self.time, self.dim))
  #   self.bestsqloss.append(np.sum(((x_tmat @ self.best_theta_hindsight[-1]).flatten() - self.y_tarr)**2)) #first term rhs of theorem 2.1 in Foster and Kakade
  #   self.best_theta_norm.append(np.linalg.norm(self.best_theta_hindsight[-1]))
  
  def update_theta_pred(self, t): #ridge regression with sherman morrision for 1 step update
    '''
    called by self.update_t()
    updates the ridge regression best theta,
    '''
    self.previnv = sherman_inv(self.previnv, self.X_dat_np[t].reshape(-1,1)) # prev inv has shape dimxdim, the second term has shape (dimx1)
    self.theta_pred = self.previnv @ self.xt_labelprodsum # theta_predarr has shape (dim,) 
    
    
  def get_ypred_t(self, t) -> None:
    '''

      This function will be run FIRST by the caller to get the ypred a scalar,
      THEN update_t(y_t) will update with the observed label and loss.
    '''
    x_t = self.X_dat_np[t] # has shape (dim,)
    yhatt = np.clip(np.dot(x_t, self.theta_pred), 0.0, 1.0) #prediction clipped to [0,1]
    self.y_predarr.append(yhatt)

  
  def update_t(self, t):
    '''
      y_t is the true label (scalar) given by environment, is in [0, 1]
      this is another external function to be called to update the model in round t
    '''
    y_t = self.y_dat_np[t] # true y, scalar
    x_t = self.X_dat_np[t] # has shape (dim,)
    # self.time += 1 #important to do this first, as the dimensions of the matrix (x_tmat) depends on it
    # self.y_tarr.append(y_t) # true label
    self.loss_tarr.append((self.y_predarr[-1] - y_t)**2) #squared loss, this is the meta experts loss to be used in outer Sleeping experts
    # self.cuml_loss_tarr.append(self.loss_tarr[-1] + self.cuml_loss_tarr[-1]) #cumulative loss
    self.xt_labelprodsum += (y_t * x_t) # has shape (dim,) 
    self.update_theta_pred(t)
    # self.update_best_hindsightt()
  
  def cleanup(self) -> None:
    pass