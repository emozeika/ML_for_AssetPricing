

import pandas as pd
import numpy as np
from datetime import datetime
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

###################################################################################################
#FUNCTION TO LOAD CSV INTO A TIMESERIES DATAFRAME
###################################################################################################

def read_ts_csv(file, format= None):
    '''
    Function to read the csv file and convert the Date column into the index of datatime type
    :param data:(file)
    :param format: (str) format the date column is in from the csv
    :return: data: (dataframe): csv file with Date column as index
    '''
    data = pd.read_csv(file, index_col= 'Date')
    if format == None:
        data.index = pd.to_datetime(data.index)
    else:
        data.index = pd.to_datetime(data.index, format=format)

    return data





#############################################################################################
#FUNCTION TO CREATE THE LAG RETURNS DF BASED ON SPECIFIED INDUSTRY
#############################################################################################

def lag_returns(data, num_lags=1):
    '''
    Function to create the lag returns of the data compared to a specified industry
    :param data: (dataframe) the data to be used in the mode
    :return: (x, y) dataframes for features and labels
    '''
    x = data.iloc[:-num_lags,:]
    y = data.iloc[num_lags:,:]
    x.index = y.index

    return x,y



############################################################################################
#FUNCTION TO TURN DATASET OF RETURNS INTO EXCESS RETURNS
############################################################################################
def subtract_rf(data, rf):

    #make sure they have the same time period
    data = data[(data.index >= rf.index[0]) & (data.index <= rf.index[-1])]
    rf  = rf[(rf.index >= data.index[0]) & (rf.index <= data.index[-1])]
    #excess returns
    data = data - rf.values

    return data


def get_momentum(x):
    # use y because it already has returns for everything
    x_temp = x.copy()
    rets = x_temp / 100 + 1
    xd = [pd.DataFrame(index=x.index) for i in range(rets.shape[1])]
    window_sizes = [i for i in range(1,13)]

    for window in window_sizes:  # however many months of momentum
        mom = pd.DataFrame(stats.zscore(rets.rolling(window=window).apply(np.prod) - 1, axis=1),
                            index=x_temp.index)  # zscore forces it into an a-range & gets rid of label=Food
        # # placeholder for alpha,beta, etc
        for j in range(mom.shape[1]):
            moms = pd.Series(mom.iloc[:, j].copy(), name=str(window) + 'm_Mom')
            xd[j] = pd.concat([xd[j], moms], axis=1)
            # placeholder for alpha,beta, etc
    for i in range(len(xd)):
        xd[i] = xd[i].dropna()

    return xd

def get_alphas_betas_ivols(x, mkt):
    mkt.index = x.index #to make sure the dates are the same
    x_temp = x.copy()
    rets = x_temp / 100 + 1
    window_sizes = [3, 6, 9, 12]
    #going to save all alphas and betas in their own dataframe based on month ie) alpha[0] = all 3 month alphas
    alphas_temp = [pd.DataFrame(index=x.index) for i in range(len(window_sizes))]
    betas_temp = [pd.DataFrame(index=x.index) for i in range(len(window_sizes))]
    ivols_temp = [pd.DataFrame(index=x.index) for i in range(len(window_sizes))]

    index = 0 #placeholder for which dataframe in alphas/betas we are on
    for window in window_sizes:
        for i in range(x_temp.shape[1]):
            #run the rolling regression to get the alpha, beta, and use them to get the errors
            exog = sm.add_constant(mkt)
            params = RollingOLS(x_temp.iloc[:,i], exog, window=window).fit(params_only=True).params  #only extracting params
            err = pd.DataFrame(x_temp.iloc[(window-1):,i] - (params.iloc[:,0] + params.iloc[:,1]*mkt.iloc[(window-1):,0]))
            #save the values into their respective dataframes
            err.columns = [str(window)+"m_ivol"]
            params.columns = [str(window) + "m_alpha", str(window) + "m_beta"]
            alphas_temp[index] = pd.concat([alphas_temp[index], params.iloc[:, 0]], axis=1)
            betas_temp[index] = pd.concat([betas_temp[index], params.iloc[:, 1]], axis=1)
            ivols_temp[index] = pd.concat([ivols_temp[index], err], axis=1)
        index += 1

    # #zscore alphas and betas
    for i in range(len(alphas_temp)):
        col_alphas, col_betas, col_ivols = alphas_temp[i].columns, betas_temp[i].columns, ivols_temp[i].columns
        alphas_temp[i] = pd.DataFrame(stats.zscore(alphas_temp[i], axis=1), index= x_temp.index)
        betas_temp[i] = pd.DataFrame(stats.zscore(betas_temp[i], axis=1), index= x_temp.index)
        ivols_temp[i] = pd.DataFrame(stats.zscore(ivols_temp[i], axis=1), index= x_temp.index)
        alphas_temp[i].columns, betas_temp[i].columns, ivols_temp[i].columns = col_alphas, col_betas, col_ivols

    #separate the alpha dataframes into their respective regressor variable
    alphas = [pd.DataFrame(index=x.index) for i in range(x_temp.shape[1])]
    betas = [pd.DataFrame(index=x.index) for i in range(x_temp.shape[1])]
    ivols = [pd.DataFrame(index=x.index) for i in range(x_temp.shape[1])]
    for i in range(len(alphas)):
        for j in range(len(alphas_temp)):
            alphas[i] = pd.concat([alphas[i], alphas_temp[j].iloc[:,i]], axis=1)
            betas[i] = pd.concat([betas[i], betas_temp[j].iloc[:,i]], axis=1)
            ivols[i] = pd.concat([ivols[i], ivols_temp[j].iloc[:,i]], axis=1)
        alphas[i] = alphas[i].dropna()
        betas[i] = betas[i].dropna()
        ivols[i] = ivols[i].dropna()

    return alphas, betas, ivols

def get_other_lags(x):
    x_temp = x.copy()
    other_lags = [pd.DataFrame(index=x_temp.index) for i in range(x_temp.shape[1])]

    for i in range(x_temp.shape[1]):
        col_names = x_temp.drop(x_temp.columns[i], axis =1).columns
        # other_lags[i] = pd.DataFrame(stats.zscore(x_temp.drop(x_temp.columns[i], axis =1), axis=1),
        #                               index=x_temp.index, columns=col_names)
        other_lags[i] = x_temp.drop(x_temp.columns[i], axis =1)

    return other_lags

def get_all_predictors(x, mkt, mkt_st):
    '''This is a function to create the following dataframes as the X matricies to use in our backtesting classes'''
    mkt.index, mkt_st.index = x.index, x.index #the days are off (nonissue since monthly data but should be fixed later
    moms = get_momentum(x)
    alphas, betas, ivols = get_alphas_betas_ivols(x, mkt)
    other_lags = get_other_lags(x)
    all_regressors = [pd.DataFrame(index = moms[0].index) for i in range(len(moms))]

    for i in range(len(moms)):
        all_regressors[i] = pd.concat([moms[i], alphas[i], betas[i], other_lags[i], mkt_st], axis = 1)
        all_regressors[i] = all_regressors[i].dropna()

    return all_regressors



