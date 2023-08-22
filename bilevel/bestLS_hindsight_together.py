# Similar to bestLS_hindsight but processes all groups together, one by one using a_t array
from river import linear_model
import numpy as np
from tqdm import tqdm

class BestLS_Hindsight_Together():
    '''
    Finds the best in hinsight at each time for a subsequence defined by a specific group
    '''
    def __init__(self, N):
        self.name = "Best LS Hindsight Together" #best least squares in hinsight, calculated online
        self.N = N
        self.experts = [linear_model.LinearRegression(l2 = 0.0) for _ in range(N)] # Least Square experts
        self.loss_experts_arr = [[] for _ in range(N)] # append losses to this
        self.best_sqloss = [] # best square loss for each group (list index has numpy array with cumulative best sq loss)
    
    def update(self, a_t, x_t, y_t):
        '''
        a_t is numpy binary array with N elements, whether group is active or not
        x_t is pandas dataframe single row
        y_t is pandas series single row
        '''
        for index, active in enumerate(a_t):
            if active:
                y_pred = self.experts[index].predict_many(x_t)
                self.loss_experts_arr[index].append((y_pred - y_t[0])**2) #squared loss that round
                self.experts[index].learn_many(x_t, y_t)
    
    def make_all_numpyarr(self):
        for gnum in range(self.N):
            self.loss_experts_arr[gnum] = np.array(self.loss_experts_arr[gnum])
    
    def cumbestsqloss(self): # tochange name to get cumulative best sq loss
        self.best_sqloss = [] # in case someone mistakenly calls this again we want a fresh calculation, and not append to already filled best sq loss array
        for gnum in range(self.N):
            self.best_sqloss.append(np.cumsum(self.loss_experts_arr[gnum]))