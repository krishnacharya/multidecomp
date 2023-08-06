from river import linear_model
import numpy as np
from tqdm import tqdm

class OnlineRidgeImplementable_alwaysactive():
    '''
    Online Ridge which is always running, this is not a meta expert,
    its implementable
    '''
    def __init__(self, X_df, y_df):
        self.name = "ORidge always active implementable" #best least squares in hinsight, calculated online
        self.model = linear_model.LinearRegression(l2 = 1.0)
        self.loss_tarr = []
        self.cumreg_groupwise_oridge = []
        self.X_df = X_df
        self.y_df = y_df
        self.T = len(X_df)
        self.fill_loss()

    def fill_loss(self):
        for t in tqdm(range(self.T)):
            y_temp_ridge = np.clip(self.model.predict_many(self.X_df.iloc[[t]]).iloc[0], 0.0, 1.0)
            self.model.learn_many(self.X_df.iloc[[t]], self.y_df.iloc[t])
            self.loss_tarr.append((y_temp_ridge - self.y_df.iloc[t][0])**2)
        del self.X_df
        del self.y_df

    def fill_subsequence_losses(self, A_tarr):
        self.cumloss_groupwise_oridge = []
        N = A_tarr.shape[1]
        loss_groupwise_oridge = []
        loss_oridge_tarr = np.array(self.loss_tarr)
        for gnum in range(N): # build cumulative loss for  on each group subsequence
            loss_groupwise_oridge.append(loss_oridge_tarr[A_tarr[:, gnum].astype(bool)]) # select those losses where group gnum active
            self.cumloss_groupwise_oridge.append(np.cumsum(loss_groupwise_oridge[-1])) #cumulative sum of the previous
   
    def fill_subsequence_regrets(self, A_tarr, bestsqloss_arr):
        '''
        mask the groups and get subsequence regret for the online ridge model which is always active
        A_tarr has shape size of (dataframe x number of groups)
        bestsqloss_arr has the best in hindsight sqloss cumulative for each group, (is a list)
        '''
        # TODO use the fill_subsequence_losses here and ensure nothing breaks!, skip this because bestsqloss_arr takes time to build
        self.cumloss_groupwise_oridge = []
        self.cumreg_groupwise_oridge = []
        N = A_tarr.shape[1]
        loss_groupwise_oridge = []
        loss_oridge_tarr = np.array(self.loss_tarr)
        for gnum in range(N): # build cumulative loss for  on each group subsequence
            loss_groupwise_oridge.append(loss_oridge_tarr[A_tarr[:, gnum].astype(bool)]) # select those losses where group gnum active
            self.cumloss_groupwise_oridge.append(np.cumsum(loss_groupwise_oridge[-1])) #cumulative sum of the previous
            self.cumreg_groupwise_oridge.append(self.cumloss_groupwise_oridge[-1] - np.array(bestsqloss_arr[gnum])) #bestsquare loss for that group subsequence still the same
