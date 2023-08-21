from river import linear_model
import numpy as np
from tqdm import tqdm
from ImplementableExpertAbstract import *

class OnlineLinearModel_alwaysactive(ImplementableExpert):
    '''
    Online Linear model (LS, ridge) which is always running, this is not a meta expert,
    its implementable
    '''
    def __init__(self, l2_pen = 1.0):
        self.name = "River linear model always active"
        self.model = linear_model.LinearRegression(l2 = l2_pen)
        self.loss_tarr = []
        self.y_predarr = []
        # self.cumreg_groupwise_oridge = []
        # self.X_df = X_df
        # self.y_df = y_df
        # self.T = len(X_df)

    def get_ypred_t(self, X_t):
        ypred = np.clip(self.model.predict_many(X_t).iloc[0], 0.0, 1.0) # ypred here is a numpy float scalar
        self.y_predarr.append(ypred)
        return ypred
    
    def update_t(self, X_t, y_t):
        self.model.learn_many(X_t, y_t)
        self.loss_tarr.append((self.y_predarr[-1] - y_t.iloc[0])**2) #ypred 
    # def fill_loss(self):
    #     for t in tqdm(range(self.T)):
    #         y_temp_ridge = np.clip(self.model.predict_many(self.X_df.iloc[[t]]).iloc[0], 0.0, 1.0)
    #         self.model.learn_many(self.X_df.iloc[[t]], self.y_df.iloc[t])
    #         self.loss_tarr.append((y_temp_ridge - self.y_df.iloc[t][0])**2)
    #     del self.X_df
    #     del self.y_df

    def fill_subsequence_losses(self, A_t):
        self.cumloss_groupwise_oridge = []
        N = A_t.shape[1]
        loss_groupwise_oridge = []
        loss_oridge_tarr = np.array(self.loss_tarr)
        for gnum in range(N): # build cumulative loss for  on each group subsequence
            loss_groupwise_oridge.append(loss_oridge_tarr[A_t[:, gnum].astype(bool)]) # select those losses where group gnum active
            self.cumloss_groupwise_oridge.append(np.cumsum(loss_groupwise_oridge[-1])) #cumulative sum of the previous
   
    # def fill_subsequence_regrets(self, A_t, bestsqloss_arr):
    #     '''
    #     mask the groups and get subsequence regret for the online ridge model which is always active
    #     A_t has shape size of (dataframe x number of groups)
    #     bestsqloss_arr has the best in hindsight sqloss cumulative for each group, (is a list)
    #     '''
    #     self.cumloss_groupwise_oridge = []
    #     self.cumreg_groupwise_oridge = []
    #     N = A_t.shape[1]
    #     loss_groupwise_oridge = []
    #     loss_oridge_tarr = np.array(self.loss_tarr)
    #     for gnum in range(N): # build cumulative loss for  on each group subsequence
    #         loss_groupwise_oridge.append(loss_oridge_tarr[A_t[:, gnum].astype(bool)]) # select those losses where group gnum active
    #         self.cumloss_groupwise_oridge.append(np.cumsum(loss_groupwise_oridge[-1])) #cumulative sum of the previous
    #         self.cumreg_groupwise_oridge.append(self.cumloss_groupwise_oridge[-1] - np.array(bestsqloss_arr[gnum])) #bestsquare loss for that group subsequence still the same
