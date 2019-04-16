# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 23:54:15 2019

@author: reblivi
"""

def time_series_signature(dataset, interval):
    
    dici = {}
    mean_one, std_one, mean_increase, mean_decrease = {}, {}, {}, {}
    max_increase, max_decrease, std_diff, max_mountain, max_valley = {}, {}, {}, {}, {}
    for ind in np.arange(0, dataset.shape[0]):
        
        dataset_interm = pd.DataFrame(data={'price': dataset.iloc[ind].values})
        mean_one_temp, std_one_temp, mean_increase_temp, mean_decrease_temp = {}, {}, {}, {}
        max_increase_temp, max_decrease_temp, std_diff_temp, max_mountain_temp, max_valley_temp = {}, {}, {}, {}, {}
        for interv in np.arange(0,dataset_interm.shape[0], interval):
            
            mean_one_temp[interv] = dataset_interm.iloc[interv:interv+interval].mean().values
            std_one_temp[interv] = dataset_interm.iloc[interv:interv+interval].std().values
            
            diff = dataset_interm.iloc[interv:interv+interval].diff()
            
            mean_increase_temp[interv] = (diff.iloc[np.where(diff > 0)[0]].mean() / len(diff.iloc[np.where(diff > 0)[0]])).values
            mean_decrease_temp[interv] = (diff.iloc[np.where(diff < 0)[0]].mean() / len(diff.iloc[np.where(diff < 0)[0]])).values  
            
            max_increase_temp[interv] = diff.iloc[np.where(diff > 0)[0]].max().values
            max_decrease_temp[interv] = diff.iloc[np.where(diff < 0)[0]].min().values
            
            std_diff_temp[interv] = (np.sqrt(np.sum(np.power((diff - diff.mean()),2)) / (interval - 1))).values
            
            mounts, valley = mounts_valleys(dataset_interm.iloc[interv:interv+interval])
            
            if len(mounts) != 0:
                max_mountain_temp[interv] = np.max(mounts)
            else:
                max_mountain_temp[interv] = np.nan
            
            if len(valley) != 0:
                max_valley_temp[interv] = np.min(valley)
            else:
                max_valley_temp[interv] = np.nan
                
        mean_one[ind] = mean_one_temp
        std_one[ind] = std_one_temp
        mean_increase[ind] = mean_increase_temp
        mean_decrease[ind] = mean_decrease_temp
        max_increase[ind] = max_increase_temp
        max_decrease[ind] = max_decrease_temp
        std_diff[ind] = std_diff_temp   
        max_mountain[ind] = max_mountain_temp
        max_valley[ind] = max_valley_temp
        
        
    return mean_one, std_one, mean_increase, mean_decrease, max_increase,max_decrease,std_diff, max_mountain, max_valley