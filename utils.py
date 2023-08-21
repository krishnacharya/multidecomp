from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder,StandardScaler, OneHotEncoder
# from sklearn.preprocessing import TargetEncoder
import pandas as pd
import pickle
def numeric_scaler(df, cols):
    '''
    df: pandas dataframe
    numeric_cols: (array of strings) column names for numeric variables

    no return: does inplace operation
    '''
    df_new = df.copy()
    mmscaler = MinMaxScaler()
    df_new[cols] = mmscaler.fit_transform(df_new[cols])
    return df_new

def ordinal_encoder(df, cols): # similar to label encoder which only works for targets?
    '''
    Encode categorical into 0 ... n-1
    '''
    df_new = df.copy()
    ordinal_enc = OrdinalEncoder()
    df_new[cols] = ordinal_enc.fit_transform(df_new[cols])
    return df_new

def oh_sklearn(df, cols):
    pass
    # Issues with this operation as it doesnt preseve number of columns etc, the dummies method below works
    # df_new = df.copy()
    # oh_enc = OneHotEncoder()
    # df_new[cols] = oh_enc.fit_transform(df_new[cols])
    # return df_new

def one_hot(df, cols): # idk if sklearns one-hot encoder is similar
    """
    df: pandas DataFrame
    param: cols a list of columns to encode 
    return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df

def target_encoder(df, x_cols, y_col):
    df_new = df.copy()
    enc_auto = TargetEncoder(smooth="auto", target_type="continuous")
    df_new[x_cols] = enc_auto.fit_transform(df_new[x_cols], df[y_col])
    return df_new

def target_encoder2(df, x_cols, y_col):
    df_new = df.copy()
    enc_auto = TargetEncoder(smooth="auto", target_type="continuous")
    df_new[x_cols] = enc_auto.fit_transform(df_new[x_cols], df[y_col])
    return df_new, enc_auto

def BaselinevsAnh(baseline_imp, Anh) -> pd.DataFrame:
    '''
        Anh has the meta experts
        baseline_imp: single linear(ridge or LS), Hoeffding tree learner which is always on
    '''
    # TODO
    columns = ['Tg', baseline_imp.name + 'loss', 'Anh cuml loss', 'relative diff %']
    results_df = pd.DataFrame(columns=columns)
    N = len(Anh.cuml_loss_curve)
    for gnum in range(len(An)):
        baseline_loss = baseline_imp.cumloss_groupwise_oridge[gnum]
        Tg = len(baseline_loss) # number of rounds this group is active
        Anh_end = Anh.cuml_loss_curve[gnum][-1] # last time steps cumulative loss
        oridge_end = oridge_loss_g[-1]
        rel_diff = 2 * abs(oridge_end - Anh_end) / (abs(oridge_end) + abs(Anh_end))
        results_df.loc[results_df.shape[0]] = [Tg, oridge_end, Anh_end, rel_diff*100]
    return results_df

def save_pickle(obj, file_dest):
    pickle_out = open(file_dest, "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()

def load_pickle(file_source):
    pickle_in = open(file_source,"rb")
    return pickle.load(pickle_in)






    

