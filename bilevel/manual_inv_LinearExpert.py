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
    self.dim = X_dat_np.shape[1] # dimensionality of the actions
    self.theta_pred = np.zeros(self.dim) # has shape (dim,)
    self.y_predarr = [] #y_hattarr, yhat t array, store the clipped [0, 1] values of the algorithm
    self.loss_tarr = [] # square loss arr for the algorithm, loss at time t (y_hatt- y_t)^2
    self.previnv = (1.0/ self.l2_pen) * np.identity(self.dim) # running A^{-1} required for ridge regression, initially set to identity, so always invertible sum of psd
    self.xt_labelprodsum = np.zeros(self.dim)
  
  def update_theta_pred(self, t) -> None: #ridge regression with sherman morrision for 1 step update
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
    # yhatt = np.clip(np.dot(x_t, self.theta_pred), 0.0, 1.0) #prediction clipped to [0,1]
    yhatt = np.dot(x_t, self.theta_pred)
    self.y_predarr.append(yhatt)

  
  def update_t(self, t) -> None:
    '''
      y_t is the true label (scalar) given by environment, is in [0, 1]
      this is another external function to be called to update the model in round t
    '''
    y_t = self.y_dat_np[t] # true y, scalar
    x_t = self.X_dat_np[t] # has shape (dim,)
    self.loss_tarr.append((self.y_predarr[-1] - y_t)**2) #squared loss, this is the meta experts loss to be used in outer Sleeping experts
    self.xt_labelprodsum += (y_t * x_t) # has shape (dim,) 
    self.update_theta_pred(t)
  
  def cleanup(self) -> None:
    self.X_dat_np = None
    self.y_dat_np = None
    self.y_predarr = np.array(self.y_predarr)
    self.loss_tarr = np.array(self.loss_tarr)


