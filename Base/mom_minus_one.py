# -*- coding: utf-8 -*-
"""
Created on Thu May 14 07:23:02 2020

@author: gth667h
"""

def get_momentum(x,w,z,inc_last):
    x_temp = x.copy()
    rets = x_temp / 100 + 1
    rets_1 = rets-1
    xd = [pd.DataFrame(index=x.index) for i in range(rets.shape[1])]
    window_sizes = [i for i in range(w,z)]
    
    for window in window_sizes:
        # however many months of momentum    
        mom = rets.rolling(window=window).apply(np.prod) - 1
        if inc_last == True:
            mom = mom    
        else:
            mom = mom-rets_1
        mom = pd.DataFrame(stats.zscore(mom, axis=1),index=x_temp.index)  # zscore forces it into an a-range & gets rid of label=Food
            
        for j in range(mom.shape[1]):
            moms = pd.Series(mom.iloc[:, j].copy(), name=str(window) + 'm_Mom')
            xd[j] = pd.concat([xd[j], moms], axis=1)
            # placeholder for alpha,beta, etc
    for i in range(len(xd)):
        xd[i] = xd[i].dropna()

    return xd