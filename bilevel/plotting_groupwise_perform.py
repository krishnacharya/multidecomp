import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bilevel.manual_inv_LinearExpert import Manual_inv_LinearExpert
from bilevel.build_all_models import *
from collections import defaultdict

class plot_groupwise:
    def __init__(self, df : pd.DataFrame, target_col:str, A_t : pd.DataFrame):
        '''
        df: dataframe minmax numeric, one hot categoric
        df must have all features then column for y
        target_col : regression target
        A_t is the group membership dataframe has shape (T x |G|+1), +1 for the always active expert
        '''
        self.target_col = target_col
        self.X_dat = df.drop(target_col, axis = 1)
        self.y_dat = df[target_col]
        self.A_t = A_t
        self.group_names = list(A_t.columns)
        self.group_sizes = list(A_t.sum(axis = 0))
        self.T = A_t.shape[0] # number of rows in dataframe
        self.N = A_t.shape[1] # number of groups
        self.rand_seeds =  [473, 503, 623, 550, 692, 989, 617, 458, 301, 205] # random seeds used to shuffle dataframe and group, to get mean & variance of cumulative loss
        
    def build_all_seeds(self, l2_pen:float = 1.0):
        '''
        Collects Anh and baseline performance for each of the shuffled dataframes
        '''
        def add_to_dic_res(b_ridgebase, b_Anh):
            cumloss_base = b_ridgebase.expert.cumloss_groupwise
            cumloss_groupwise_ada = b_Anh.Anh.cumloss_groupwise_ada
            for g_ind, gname in enumerate(self.group_names):
                base = cumloss_base[g_ind][-1]
                ada = cumloss_groupwise_ada[g_ind][-1]
                self.dic_res_base[gname].append(base)
                self.dic_res_Anh[gname].append(ada)
        
        # this dictionary has results of base, Anh across different shuffles
        self.dic_res_base = defaultdict(list)
        self.dic_res_Anh = defaultdict(list)
        for seed in self.rand_seeds:
            X_shuf = self.X_dat.sample(frac = 1, random_state = seed)
            y_shuf = self.y_dat.sample(frac = 1, random_state = seed)
            A_t_shuf_np = self.A_t.sample(frac = 1, random_state = seed).to_numpy()
            X_dat_np = X_shuf.to_numpy()
            y_dat_np = y_shuf.to_numpy()

            dirname_base = './models_adult/baseline/'
            filename = 'manual_ridge_seed='+ str(seed)+ ' '
            b_ridgebase = build_baseline_alwayson(dirname_base, filename, \
                          A_t_shuf_np, Manual_inv_LinearExpert(X_dat_np, y_dat_np, l2_pen = l2_pen))
            dirname_Anh = './models_adult/Anh/'
            experts = [Manual_inv_LinearExpert(X_dat_np, y_dat_np, l2_pen = l2_pen) for _ in range(self.N)]
            b_Anh = build_Anh(dirname_Anh, filename, A_t_shuf_np, experts)
            add_to_dic_res(b_ridgebase, b_Anh)
    
    def build_df_res(self):
        '''
        builds the Anh and baseline results dataframe, 
        '''
        self.df_res_base, self.df_res_Anh = pd.DataFrame(self.dic_res_base), pd.DataFrame(self.dic_res_Anh) #each row has cumulative loss of each group in a rand seed run
        self.df_base_meansd = self.df_res_base.describe().loc[['mean', 'std']].T # mean and sd are the columns
        self.df_Anh_meansd = self.df_res_Anh.describe().loc[['mean', 'std']].T
       
        self.df_base_meansd.rename(columns={'mean': 'mean_base', 'std': 'std_base'}, inplace=True)
        self.df_Anh_meansd.rename(columns={'mean': 'mean_Anh', 'std': 'std_Anh'}, inplace=True)

    def plot_save_subgroups(self, subgroups_list : list):
        '''
        subgroups_list [[young, middle, old], [male, female], ...] list of all the atmoic in each subgroup
        '''
        for subgroups in subgroups_list:
            df_base_sg = self.df_base_meansd.loc[subgroups]
            df_Anh_sg = self.df_Anh_meansd.loc[subgroups]
            group_bar_plot_df = pd.concat([df_base_sg, df_Anh_sg], axis = 1)
            yerr = group_bar_plot_df[['std_base', 'std_Anh']].to_numpy().T
            group_bar_plot_df[['mean_base', 'mean_Anh']].plot(kind='bar', yerr=yerr, alpha=0.9, error_kw=dict(ecolor='k'))
            plt.show()