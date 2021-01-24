# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:32:02 2020

@author: lfiorentini
version: 1.1

"""
import numpy as np
from datetime import datetime
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics.pairwise import euclidean_distances

def remove_not_common(Data, common_head):
    for i in Data.columns:
        if i not in common_head:
            Data.drop(i, axis = 1, inplace = True)   

hour_format = '%m/%d/%Y %I:%M:%S %p'
def working_day(date_str):
    try:
        day_code = datetime.strptime(date_str, hour_format)
    except ValueError:
        print('StrError:', date_str)
    day_code = day_code.weekday()
    return day_code != 0 and day_code != 6

def moment_day(date_str):
    hour = datetime.strptime(date_str, hour_format).hour
    if hour <= 6 or hour >= 23:
        return "Night"
    if hour <= 10 or hour >= 7:
        return "Morning"
    if hour <= 18 or hour >= 11:
        return "Working"
    return "Evening"

def get_timestamp(date_str):
    return datetime.timestamp(datetime.strptime(date_str, hour_format))

def tequila_post(date_str):
    post_date = datetime.strptime(date_str, hour_format)
    if post_date.year >= 2019:
        if post_date.year > 2019 or post_date.month >= 5:
            return 1
    return 0

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return [m-h, m, m+h]

def intersection_confidence_interval(int_1, int_2):
    new_min = max(int_1[0],int_2[0])
    new_max = min(int_1[2],int_2[2])
    return [new_min, new_max]

def test_same_mean(data_1, data_2, confidence=0.95):
    int_1 = mean_confidence_interval(data_1, confidence=confidence)
    int_2 = mean_confidence_interval(data_2, confidence=confidence)
    new_int = intersection_confidence_interval(int_1, int_2)
    return new_int[0] < new_int[1]
# function that extract from a dataset only the elements between two dates
def campaign_selection(dataset, first_day, last_day):
    first_timestamp = datetime.timestamp(datetime.strptime(first_day, '%m/%d/%Y %I:%M:%S %p'))
    last_timestamp = datetime.timestamp(datetime.strptime(last_day, '%m/%d/%Y %I:%M:%S %p'))
    res = dataset[(dataset['timestamp'] >= first_timestamp) & (dataset['timestamp'] <= last_timestamp)].copy()
    return res

def count_key_message(Message, key_set):
    res = 0
    for word in key_set:
        if word in Message:
            res += 1
    return res

def paid_efficacity(org, paid):
    if org != 0:
        return paid/org
    return float(paid > 0)

def KPI_dict_FB(data, with_marge, confidence = 0.95):
    index_IC = 1
    if with_marge:
        index_IC = 0
    dict_FB = {
       'Total post': data.shape[0],
        'Total video': sum(data['Type_Video'] == 1),
        'Video/post': sum(data['Type_Video'] == 1) / data.shape[0],
        'Average total reach / post': mean_confidence_interval(data['Post Total Reach'],
                                                        confidence = confidence)[index_IC],
        'Average organic reach / post': mean_confidence_interval(data['Post organic reach'],
                                                        confidence = confidence)[index_IC],
        'Average video total reach / post': mean_confidence_interval(data[data['Type_Video'] == 1]['Post Total Reach'],
                                                        confidence = confidence)[index_IC],
        'Average video organic reach / post': mean_confidence_interval(data[data['Type_Video'] == 1]['Post organic reach'],
                                                        confidence = confidence)[index_IC],
        'Organic Video Views / post': mean_confidence_interval(data[data['Type_Video'] == 1]['Organic Video Views'],
                                                               confidence = confidence)[index_IC],
        'Paid Video Views / post': mean_confidence_interval(data[data['Type_Video'] == 1]['Paid Video Views'],
                                                               confidence = confidence)[index_IC],
        'Videos Average views (>=30s)': sum((data['Type_Video'] == 1) & (data['Average time video viewed'] >= 30))/
            sum(data['Type_Video'] == 1),
        'Average share / post': mean_confidence_interval(data['Talking About This (Post) by action type - share'],
                                                        confidence = confidence)[index_IC],
        'Average like / post': mean_confidence_interval(data['Talking About This (Post) by action type - like'],
                                                       confidence = confidence)[index_IC],
        'Average comment / post': mean_confidence_interval(data['Talking About This (Post) by action type - comment'],
                                                          confidence = confidence)[index_IC],
        'Videos Average share / post':
         mean_confidence_interval(data[data['Type_Video'] == 1]['Talking About This (Post) by action type - share'],
                                 confidence = confidence)[index_IC],
        'Videos Average like / post':
        mean_confidence_interval(data[data['Type_Video'] == 1]['Talking About This (Post) by action type - like'],
                                confidence = confidence)[index_IC],
        'Videos Average comment / post':
        mean_confidence_interval(data[data['Type_Video'] == 1]['Talking About This (Post) by action type - comment'],
                                confidence = confidence)[index_IC],
        'Videos Average Completion (percentage)':mean_confidence_interval(
        data[data['Type_Video'] == 1]['Average time video viewed']/data[data['Type_Video'] == 1]['Video length'],
        confidence = confidence)[index_IC]*100
    }
    for k in dict_FB.keys():
        print(k, dict_FB[k])
    return dict_FB

class Iterative_ANOVA():
    def __init__(self, data, y_lab, x_in_lab_list, thres = 0.01):
        self.data = data
        self.thres = thres
        self.y_lab = y_lab
        self.__x_lab_list__ = []
        for lab in x_in_lab_list:
            self.__x_lab_list__.append('C(' + lab + ')')
        self.__full_list__ = self.__x_lab_list__ + self.__create_inter__(self.__x_lab_list__)
        self.__curr_list__ = self.__full_list__
        self.p_values_history = {key : [] for key in self.__full_list__}
        self.history = []
        self.last_model = []
        
    def __create_inter__(self, __x_lab_list__):
        res = []
        for i in range(len(__x_lab_list__)-1):
            for j in range(i+1,len(__x_lab_list__)):
                res.append(__x_lab_list__[i] + ':' + __x_lab_list__[j])
        return res
    
    def __create_formula__(self):
        formula = self.y_lab + ' ~ '
        for lab in self.__curr_list__:
            formula = formula + lab + ' + '
        return formula[:-3]
    
    # function that remove the interaction terms of labels already used 
    def __remove_inter__(self, basic_label):
        to_remove = []
        for lab in self.__curr_list__:
            if basic_label in lab:
                to_remove.append(lab)
        
        for lab in to_remove:
            self.__curr_list__.remove(lab)
            
    def fit(self):
        max_p = 1
        max_lab = 'not_a_label'
        curr_anova = []
        n_iter = 0
        print("Starting ANOVA with:")
        print(self.__curr_list__)
        while max_p > self.thres:
            #remove feature and its interactions
            if max_lab in self.__curr_list__:
                    self.__remove_inter__(max_lab)
            
            #create new formula
            curr_formula = self.__create_formula__()
            
            #compute new ANOVA
            curr_anova = ols(curr_formula, data = self.data).fit()
            table = sm.stats.anova_lm(curr_anova, typ=2) # Type 2 ANOVA DataFrame
            max_lab = table['PR(>F)'].idxmax()
            max_p = table['PR(>F)'].max()
            print("ANOVA iter", n_iter)
            print(self.__curr_list__)
            print(max_lab)
            print(max_p)
            self.history.append(curr_anova)
            for lab in self.__full_list__:
                if lab in self.__curr_list__:
                    self.p_values_history[lab].append(table['PR(>F)'][lab])
                else:
                    self.p_values_history[lab].append(1)
        
            n_iter += 1
        
        self.last_model = curr_anova
        return curr_anova

    def get_curr_list(self):
        return self.__curr_list__
    
    def show_p_values_history(self):
        return self.p_values_history
    
    def summary(self):
        print(self.show_p_values_history())
        return self.last_model

def mean_intra_cluster_dist(clusters, distance = euclidean_distances):
    mean = 0
    n_clusters = clusters.shape[0]
    n_feat = clusters.shape[1]
    for i in range(n_clusters-1):
        for j in range(i+1,n_clusters):
            mean += distance(clusters[i,:].reshape(1, n_feat), clusters[j,:].reshape(1, n_feat))
    return float(mean/(n_clusters*(n_clusters-1)/2))