import pandas as pd
from utils import *
import numpy as np
import joblib
import time
from lean_adahedge import *
from OnlineLinExpert import *
from OnlineLinModel_alwaysactive_implementable import *

class build_Anh: #maybe make an abstract class for building models
    def __init__(self, dir_name : str, filename: str, X_dat: pd.DataFrame, y_dat: pd.DataFrame, \
                A_t: np.ndarray, cat_cols_sig: list[str], groups : list[str], experts: list[Expert]):
        '''
        Assuming that the dataframe has been processed already (make dataframe management class pipeline)

        groups:
            list of sensitive groups, e.g. 2 sexes, 9 races
        experts: 
            list of meta experts, could be linear(ridge, LS), trees etc
        '''
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir_name = dir_name
        self.filename = filename + self.timestamp
        self.X_dat = X_dat
        self.y_dat = y_dat
        self.A_t = A_t
        self.T = A_t.shape[0]
        self.cat_cols_sig = cat_cols_sig
        self.groups = groups
        self.experts = experts
        self.build()

    def build(self):
        Anh = Adanormal_sleepingexps(self.N, self.experts) #adanormal hedge
        for t in tqdm(range(self.T)):
            Anh.get_prob_over_experts(self.A_t[t]) #get probability over meta-experts
            Anh.update_metaexps_loss(self.A_t[t], self.X_dat.iloc[[t]], self.y_dat.iloc[[t]]) # update internal states of the meta-experts
            Anh.build_cumloss_curve(self.A_t)
            Anh.cleanup_for_saving() #compact size after cleanup, only essential external varaibles saved
            save_loc = f'''./{self.dir_name}/models/Anh_sepbuild/{self.filename}.pkl'''
            joblib.dump(Anh, save_loc)

class build_baseline_alwayson:
    def __init__(self, dir_name : str, filename: str, X_dat: pd.DataFrame, y_dat: pd.DataFrame, \
                A_t: np.ndarray, implementable_expert: ImplementableExpert, l2_pen = 1.0):
        '''
            implementable expert function class matches the one in Anh meta experts
            for e.g. if least squares as baseline, then Anh meta experts are also LS
        '''
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir_name = dir_name
        self.filename = filename + self.timestamp
        self.X_dat = X_dat
        self.y_dat = y_dat
        self.A_t = A_t 
        self.T = A_t.shape[0]
        # self.cat_cols_sig = cat_cols_sig
        # self.groups = groups
        self.l2_pen = l2_pen
        self.implementable_expert = implementable_expert
        self.build()
    
    def build(self):
        for t in tqdm(range(self.T)):
            self.implementable_expert.get_ypred_t(self.X_dat.iloc[[t]])
            self.implementable_expert.update_t(self.X_dat.iloc[[t]], self.y_dat.iloc[[t]])
        self.implementable_expert.fill_subsequence_losses(self.A_t) # finds losses on masked subsequences for each group
        save_loc = f'''./{self.dir_name}/models/linear_sepbuild/{self.filename}_l2pen={self.l2_pen}.pkl'''
        joblib.dump(self.implementable_expert, save_loc)

class All_linear_models: # can make it an abstract class that extends from all_models, but TODO for later
    '''
        TODO add class documentation
    '''
    def __init__(self, dir_name : str, filename: str, df_oh: pd.DataFrame, cat_cols_sig: list, groups : list):
        '''
            dir_name: name of head of directory, e.g. ./one_hotencoded
            filename: name of file
            df_oh: the dataframe with income <= 200k, no other processing done, possibly sparsified
            cat_cols_sig: list with names of all the categorical columns to be one-hot encoded
            groups: all the sensitive groups with underscore _ , e.g. ['SEX_1','SEX_2', 'RAC1P_1'....]
        '''
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir_name = dir_name
        self.filename = filename + self.timestamp
        self.df_oh = df_oh
        self.cat_cols_sig = cat_cols_sig
        self.groups = groups
        
    def build_models(self, to_drop_groups = False, \
                    di_to_fill: dict = {'A_t': None, 'bls': None, 'Anh': None, 'oridge_implementable': None},\
                    to_shuffle = False, seed = 21) -> None:
        '''
            Goal:
                a) Saves numpy array (A_t) which is binary and has shape (T x len(sens_groups))
                b) Saves best squared loss (bls) in hindsight object
                c) Saves Ada normal hedge (Anh) with ridge meta experts
                d) Saved Online Ridge implementable, oridge_implementable, this is a single online ridge, regrets on subsequences computed by masking

            Input:

                di_to_fill : A_t, bls, Anh, oridge_implementable are added to dictionary
                to_drop_groups: Whether or not to drop sensitive columns
                to_shuffle: Whether or not to shuffle the dataframe

            Also Fills into di_to_fill dictioanry:
            A_t: numpy array
            bls: bestLS_hindsight.BestLS_Hindsight object
            Anh: Adanormal_sleepingexps.Adanormal_sleepingexps object
            oridge_implementable: oridge_alwaysactive_implementable.OnlineRidgeImplementable_alwaysactive object
        '''
        def build_bls(): # best square loss used to compute regret
            # TODO skip this, long to build, just compare squares losses sum
            bls = BestLS_Hindsight_Together(self.N)
            for t in tqdm(range(self.T)):
                bls.update(A_t[t], X_dat.iloc[[t]], y_dat.iloc[t])
            bls.make_all_numpyarr() 
            bls.cumbestsqloss() # build cumulative loss of least squares on each group
            # di_to_fill['bls'] = bls
            joblib.dump(bls, self.dir_name + 'models/bestsqloss/'+ self.filename +'.pkl')

        def build_Anh():
            experts = [River_OnlineRidge() for _ in range(self.N)] # Online ridge meta-experts
            Anh = Adanormal_sleepingexps(self.N, experts) #adanormal hedge
            for t in tqdm(range(self.T)):
                Anh.get_prob_over_experts(A_t[t]) #get probability over meta-experts
                Anh.update_metaexps_loss(A_t[t], X_dat.iloc[[t]], y_dat.iloc[t]) # update internal states of the meta-experts
            # fill in Anh cumulative regret curve before saving, to reduce disk space
            # Anh.build_cumloss_curve(di_to_fill['bls'].best_sqloss, A_t)
            Anh.build_cumloss_curve(A_t)
            Anh.cleanup_for_saving() #compact size after cleanup, only essential external varaibles saved
            # di_to_fill['Anh'] = Anh
            joblib.dump(Anh, self.dir_name + 'models/Anh/'+ self.filename +'.pkl')
        
        def build_oridge_implementable():
            oridge_implementable = OnlineRidgeImplementable_alwaysactive(X_dat, y_dat)
            oridge_implementable.fill_subsequence_losses(A_t)
            # oridge_implementable.fill_subsequence_regrets(A_t, di_to_fill['bls'].best_sqloss)
            # di_to_fill['oridge_implementable'] = oridge_implementable
            joblib.dump(oridge_implementable, self.dir_name + 'models/oridge_implementable/'+ self.filename + '.pkl')

        # first one-hot encode self.data_fil_oh using cat_cols_sig
        # df_oh = one_hot(self.data_fil, self.cat_cols_sig)
        # df_oh.drop(self.cat_cols_sig, axis=1, inplace = True) # drop OCCP, RACE, SEX, MAR..all categorical
        # shuffle if said to do so
        if to_shuffle:
            self.filename = self.filename + "_shuffled_"
            self.df_oh = self.df_oh.sample(frac = 1, random_state = seed)
        else:
            self.filename = self.filename + "_unshuffled_"
            
        # now collect build the A_t numpy array before dropping those columns (which can happen if dropped = True)
        A_tdf = self.df_oh[self.groups]
        A_tdf['alwayson'] = 1 # adds the always active / using all data "group"
        A_t = A_tdf.to_numpy()
        # di_to_fill['A_t'] = A_t
        self.T  = A_t.shape[0] # shape of A_t is T x len(number of sens groups) + 1
        self.N  = A_t.shape[1]
        if to_drop_groups:
            self.filename = self.filename + "_dropped_"
            self.df_oh.drop(self.groups, axis = 1, inplace=True) # drop the onehot SEX_1, SEX_2, RAC1P_1,...
        else:
            self.filename = self.filename + "_undropped_"

        np.save(self.dir_name + 'nparrays/' + self.filename , A_t) # save the A_t array on disk

        # now min max scale all the features, all in [0, 1], and 
        # self.df_oh = numeric_scaler(self.df_oh, self.df_oh.columns)
        X_dat = self.df_oh.drop('PINCP', axis=1) # dropping the income column
        y_dat = pd.DataFrame(self.df_oh['PINCP']) 
        # build_bls()
        build_oridge_implementable()
        build_Anh()

