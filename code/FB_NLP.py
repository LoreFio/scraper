# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:07:55 2020

@author: lfiorentini
version: 1.1
"""

import re
import numpy as np
from os.path import join as path_join# import system for path definition
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set(rc={'figure.figsize':(6,4)})

import pickle
from text_processer import Text_processer
from noise_filter import Noise_filter
from global_var_nlp import output_dir, sub_dict, audiences
from langdetect import detect
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

def process_message(message, cur_filter, processer):
    filtered = cur_filter.filter_re([message])[0]
    tokens = processer.tokenize(filtered)
    lems = processer.word_normalization(processer.remove_numbers(tokens))
    return lems

def len_inter(list1, list2):
    return len(set(list1) & set(list2))

def NLP_processed(data, text_col, cur_filter, processer, result, aud):
    return data.merge(data[text_col].apply(lambda s: pd.Series({aud:s+1, 'feature2':s-1})), 
        left_index=True, right_index=True)

key_dir = "../output/all_sites"
language = "english"
    
if language == 'english':
    normalizer = "WNLemmatizer"
elif language == 'french':
    normalizer = "SnowballStemmer"

study_paid = True

FB_filter = Noise_filter(sub_dict)
processer = Text_processer(using_re_token = True, normalizer = normalizer,
                           language = language)

NLP_data = pd.read_csv(path_join(output_dir, 'FB_data.csv'))
NLP_data['language'] = NLP_data['Post Message'].apply(detect)
french_post = NLP_data[NLP_data['language'] == 'fr']['Post Message'].head()
italian_post = NLP_data[NLP_data['language'] == 'it']['Post Message'].head()
NLP_data = NLP_data[NLP_data['language'] == 'en']

NLP_data['processed'] = NLP_data['Post Message'].apply(process_message,
                                                       cur_filter = FB_filter,
                                                       processer = processer)
NLP_data['useful_words'] = NLP_data['processed'].apply(len) 
NLP_data['char_len'] = NLP_data['Post Message'].apply(len)
NLP_data['is_paid'] = NLP_data['is_paid'].apply(int)
NLP_data.head()

pkl_ext = '.pkl'

for aud in audiences:
    with open(path_join(key_dir, aud + pkl_ext), 'rb') as inp:
        result = pickle.load(inp)
        inp.close()
    #NLP_data[aud + '_all'] = NLP_data['processed'].apply(len_inter, list2 = result.all_k)
    NLP_data[aud + '_all_f'] =\
        NLP_data['processed'].apply(len_inter, list2 = result.all_k) / \
            NLP_data['useful_words']
    #NLP_data[aud + '_joint'] = NLP_data['processed'].apply(len_inter, list2 = result.joint_k)
    NLP_data[aud + '_joint_f'] =\
        NLP_data['processed'].apply(len_inter, list2 = result.joint_k) / \
            NLP_data['useful_words']
    for key in result.dict_k:
        if len(result.dict_k[key]) > 0:
            #NLP_data[aud + '_' + key] = NLP_data['processed'].apply(len_inter, list2 = result.dict_k[key])
            NLP_data[aud + '_' + key + '_f'] = \
                NLP_data['processed'].apply(len_inter,
                                            list2 = result.dict_k[key]) / \
                    NLP_data['useful_words']

Message_data = NLP_data[['Post Message', 'processed']]
NLP_data.drop(['Post Message', 'processed', 'language'], axis = 1,
              inplace = True)

NLP_data = NLP_data.loc[:,NLP_data.apply(pd.Series.nunique) != 1]
Q_NLP = NLP_data.corr()
"""
pict_NLP = "nlp.html"
Html_file= open(path_join(directory_img, pict_NLP),"w")
Html_file.write(Q_NLP.style.background_gradient(cmap='coolwarm').render())
Html_file.close()
sns.heatmap(Q_NLP, cmap="YlGnBu", vmin=-1, vmax=1, xticklabels = True,
            yticklabels = True)
"""

corr_data = pd.DataFrame(
    Q_NLP[Q_NLP['Post Total Reach'] >= 0.4]['Post Total Reach'])
corr_data['mean'] = NLP_data.mean()

col_data = pd.DataFrame(NLP_data.columns, columns = ['col'])
col_data = col_data.merge(
    col_data['col'].apply(
        lambda s: pd.Series({'lb':NLP_data[s].mean() - NLP_data[s].std(),
                             'mean':NLP_data[s].mean(),
                             'ub':NLP_data[s].mean() + NLP_data[s].std()})),
    left_index=True, right_index=True)
col_data.to_csv(path_join(output_dir, 'col_data.csv'), index=False, sep = ';')

'''
#test to show that the frequencies and the counters are not exactly the same
#however the difference is small and we will therefore consider only one of the
#two for each variable
for aud in audiences:
    for key in result.dict_k:
        if aud + '_' + key in Q_NLP.columns:
            diff = Q_NLP[aud + '_' + key] - Q_NLP[aud + '_' + key + '_f']
            print(aud, key, diff.mean(), diff.std())
        else:
            print(aud, key, "is useless")
'''           

if study_paid:
    objective_col = 'Post Total Reach'
    NLP_data.drop(['Post organic reach', 'Engaged Users', 'is_tequila'],
                  axis = 1, inplace = True)
else:
    objective_col = 'Post organic reach'
    NLP_data = NLP_data[NLP_data['is_paid'] == 0]
    NLP_data.drop(['Post Total Reach', 'Engaged Users', 'is_tequila'],
                  axis = 1, inplace = True)

scaler = MinMaxScaler().fit(NLP_data)
NLP_scaled = pd.DataFrame(scaler.transform(NLP_data),
                          columns = NLP_data.columns)

criterion = 'mse'
msk = np.random.rand(len(NLP_scaled)) < 0.8

train = NLP_scaled[msk]
test = NLP_scaled[~msk]
x_train = train.loc[:, train.columns != objective_col].to_numpy()
x_test = test.loc[:, test.columns != objective_col].to_numpy()
y_train =  train[objective_col].to_numpy()
y_test =  test[objective_col].to_numpy()

LR = LinearRegression()
LR.fit(x_train, y_train)
LR_pred = LR.predict(x_test)
LR_Rq_score = LR.score(x_test , y_test)


n_estimators = 100
RFR = RandomForestRegressor(n_estimators = n_estimators, criterion = criterion)
RFR.fit(x_train, y_train)
RFR_pred = RFR.predict(x_test)
RFR_Rq_score = RFR.score(x_test , y_test)
RFR_importance = RFR.feature_importances_

"""
Return the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of
squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares
((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it
can be negative (because the model can be arbitrarily worse). A constant model
that always predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.
"""

q_thres = 0.75
"""
plt.plot(NLP_scaled[objective_col], 'r')
plt.axhline(y = np.quantile(y_train, q_thres), color = 'k')
plt.ylabel('Reach scaled')
plt.xlabel('Post index')
plt.title(objective_col + ' (all post)')
plt.legend(('RF prevision', 'Separator'))
"""

plt.plot(RFR_pred, 'b')
plt.axhline(y = np.quantile(y_train, q_thres), color = 'k')
plt.plot(y_test, 'r')
plt.title('RF')
plt.ylabel('Reach scaled')
plt.xlabel('Post index')
plt.title(objective_col + ' (random post)')
plt.legend(('RF prevision', 'Separator', 'Real value'))

y_train_C = y_train >np.quantile(y_train, q_thres)
y_test_C = y_test >np.quantile(y_train, q_thres)

n_estimators = 100
RFC = RandomForestClassifier(n_estimators = n_estimators)
RFC.fit(x_train, y_train_C)
RFC_pred = RFC.predict(x_test)
RFC_Rq_score = RFC.score(x_test , y_test_C)
RFC_importance = RFC.feature_importances_


Importance_df = pd.DataFrame(columns = ['col', 'RFR_imp', 'RFC_imp'])
for i in range(1, len(train.columns)):
     Importance_df = Importance_df.append({'col': train.columns[i],
                                           'RFR_imp': RFR_importance[i-1],
                                           'RFC_imp': RFC_importance[i-1]},
                                          ignore_index = True)
     

ads_new = pd.read_csv('../../Downloads/Groupe-Renault-Ads-Lifetime_new.csv')

def clean_ads_pub(message):
    output = re.sub(r'Publication[\xa0 ]: «[\xa0 ]','',message)
    return re.sub(r'(\.\.\.)*[\xa0 ]\».*$','',output).lower()

def tr_budget(message, ads):
    budget = 0
    for i in range(len(ads)):
        if len(ads['content'][i]) > 0 and ads['content'][i] in message:
            budget += ads['Ad set budget'][i]
    return budget

ads_new['content'] = ads_new['Ad name'].apply(clean_ads_pub)
Message_data['extra'] = Message_data['Post Message'].apply(tr_budget,
                                                           ads = ads_new)
