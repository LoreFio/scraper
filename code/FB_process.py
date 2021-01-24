# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:29:51 2020

@author: lfiorentini
version: 1.1
"""

import numpy as np
from os.path import join as path_join# import system for path definition
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from FB_utilities import get_timestamp, hour_format, moment_day, \
    remove_not_common, tequila_post, working_day, mean_confidence_interval, \
        test_same_mean, campaign_selection,\
            KPI_dict_FB, Iterative_ANOVA, count_key_message, paid_efficacity, \
                mean_intra_cluster_dist
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.stats import shapiro
from scipy.stats import boxcox
import statsmodels.api as sm
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import sys
sns.set(rc={'figure.figsize':(6,4)})

directory = "C:/Users/lfiorentini/TequilaRapido/Renault/Data/FB"
directory_img = "C:/Users/lfiorentini/TequilaRapido/Renault/Data/FB/img"
directory_csv = "C:/Users/lfiorentini/TequilaRapido/Renault/Data/FB/csv"
output_dir = "../output"

file1 = "Apr_Aug_18.csv"
file2 = "Sep_Jan_18.csv"
file3 = "Feb_Jun_19.csv"
file4 = "Jul_Nov_19.csv"
file5 = "Dec_Mar_20.csv"
feat_str = 8
save_fig = False
reduce_only = True

headers = pd.read_csv(path_join(directory, file1), header=0, nrows = 1)#, converters = conver_dict)
type_dict = {}
for i in range(len(headers.columns)):
    if i < feat_str: 
        type_dict[headers.columns[i]] = str
    else:
        type_dict[headers.columns[i]] = np.float64 #we need float to deal with all the NaN
Data1 = pd.read_csv(path_join(directory, file1), header=0, skiprows = [1], dtype = type_dict)
Data2 = pd.read_csv(path_join(directory, file2), header=0, skiprows = [1], dtype = type_dict)
Data3 = pd.read_csv(path_join(directory, file3), header=0, skiprows = [1], dtype = type_dict)
Data4 = pd.read_csv(path_join(directory, file4), header=0, skiprows = [1], dtype = type_dict)
Data5 = pd.read_csv(path_join(directory, file5), header=0, skiprows = [1], dtype = type_dict)

Data1.dropna(axis = 1, how='all', inplace=True)
Data2.dropna(axis = 1, how='all', inplace=True)
Data3.dropna(axis = 1, how='all', inplace=True)
Data4.dropna(axis = 1, how='all', inplace=True)
Data5.dropna(axis = 1, how='all', inplace=True)

common_head = list(set(Data1.columns) & set(Data2.columns) &
                   set(Data3.columns) & set(Data4.columns) &
                   set(Data5.columns))
len(common_head)

remove_not_common(Data1, common_head)
remove_not_common(Data2, common_head)
remove_not_common(Data3, common_head)
remove_not_common(Data4, common_head)
remove_not_common(Data5, common_head)

Data_f = pd.concat([Data1, Data2, Data3, Data4, Data5], sort=True)
Data_f.reset_index(drop = True, inplace = True)

Data_f.fillna(value = 0, inplace = True)
list_useless = []
for col_name in Data_f.columns:
    if len(Data_f[col_name].unique()) <= 1:
        list_useless.append(col_name)
Data_f.drop(list_useless, axis=1, inplace=True)

Data_f.drop(['Post ID','Permalink'], axis=1, inplace=True)
#Data_f.drop(['Identifiant de la publication','Lien permanent','Publié'], axis=1, inplace=True)

#Data_f['Message de publication'] = Data_f['Message de publication'].str.lower()
Data_f['Post Message'] = Data_f['Post Message'].str.lower()
    
tmp = Data_f.columns.copy()
tmp = [x.replace("Lifetime ","") for x in tmp]
Data_f.columns = tmp

Data_f.dropna(axis = 0, how='any', inplace=True)
Data_f.drop(Data_f[Data_f['Type'] == 'SharedVideo'].index, inplace=True)
Data_f.drop(Data_f[Data_f['Type'] == 0].index, inplace=True)
Data_f.drop(Data_f[Data_f['Type'] == 'Status'].index, inplace=True)
Data_f.drop(Data_f[Data_f['Type'] == 'Link'].index, inplace=True)

timeline = Data_f[['Post Total Reach','Posted']].copy()

Data_f['timestamp'] = Data_f['Posted'].apply(get_timestamp)
Data_f['is_tequila'] = Data_f['Posted'].apply(tequila_post)
Data_f['working_day'] = Data_f['Posted'].apply(working_day)
Data_f['moment_day'] = Data_f['Posted'].apply(moment_day)
Data_f.drop('Posted', axis=1, inplace=True)
Data_f['is_paid'] = Data_f['Post Paid Reach'] > 0
Data_f = pd.get_dummies(Data_f, columns = ['moment_day', 'Type'],
                        drop_first = True)

Q1 = Data_f[Data_f['Type_Video'] == 1].drop('Type_Video', axis = 1).corr()
#for the photo we have also to remove all the features that make sense only for a video
video_attr_list = ['Type_Video', 'Organic views to 95%',
                   'Organic views to 95%.1', 'Paid views to 95%',
                   'Paid views to 95%.1', 'Organic Video Views',
                   'Organic Video Views.1', 'Paid Video Views',
                   'Paid Video Views.1', 'Organic Video Views',
                   'Organic Video Views.1', 'Average time video viewed',
                   'Video length']
Q2 = Data_f[Data_f['Type_Video'] == 1].drop(video_attr_list, axis = 1).corr()
Q3 = Data_f.corr()

pict1_html = 'Video_corr.html'
pict2_html = 'Photo_corr.html'
pict3_html = 'Global_corr.html'

if save_fig:
    Html_file= open(path_join(directory_img, pict1_html),"w")
    Html_file.write(Q1.style.background_gradient(cmap='coolwarm').render())
    Html_file.close()
    
    Html_file= open(path_join(directory_img, pict2_html),"w")
    Html_file.write(Q2.style.background_gradient(cmap='coolwarm').render())
    Html_file.close()
    
    Html_file= open(path_join(directory_img, pict3_html),"w")
    Html_file.write(Q3.style.background_gradient(cmap='coolwarm').render())
    Html_file.close()

"""Data to be removed cause highly correlated to Engaged Users"""
Data_f.drop('Matched Audience Targeting Consumers on Post', axis = 1, inplace = True)
Data_f.drop('Matched Audience Targeting Consumptions on Post', axis = 1, inplace = True)

"""Data to be removed cause highly correlated to Post Total Reach"""
Data_f.drop('Post Paid Impressions', axis = 1, inplace = True)
Data_f.drop('Post Total Impressions', axis = 1, inplace = True)
Data_f.drop('Post Paid Reach', axis = 1, inplace = True)

"""Data to be removed cause highly correlated to Post organic reach"""
Data_f.drop('Post Organic Impressions', axis = 1, inplace = True)

"""Data to be removed cause highly correlated to other features"""
Data_f.drop('Organic views to 95%.1', axis = 1, inplace = True) # we keep "Organic views to 95%"
Data_f.drop('Paid views to 95%.1', axis = 1, inplace = True) # we keep "Paid views to 95%"

Data_f.drop('Organic Video Views.1', axis = 1, inplace = True) # we keep "Organic Video Views"
Data_f.drop('Paid Video Views.1', axis = 1, inplace = True) # we keep "Paid Video Views"

Data_f.drop('Post Paid Impressions by people who have liked your Page', axis = 1, inplace = True)
# we keep "Post Impressions by people who have liked your Page"

Data_f.drop('Negative Feedback from Users', axis = 1, inplace = True)
# we keep "Negative Feedback"
Data_f.drop('Negative Feedback from Users by Type - hide_all_clicks', axis = 1, inplace = True)
# we keep "Negative Feedback by Type - hide_all_click"
Data_f.drop('Negative Feedback from Users by Type - hide_clicks', axis = 1, inplace = True)
# we keep "Negative Feedback by Type - hide_clicks"

Data_f.drop('Post Stories by action type - share', axis = 1, inplace = True)
# we keep "Talking About This (Post) by action type - share"
Data_f.drop('Post Stories by action type - like', axis = 1, inplace = True)
# we keep "Talking About This (Post) by action type - like"
Data_f.drop('Post Stories by action type - comment', axis = 1, inplace = True)
# we keep "Talking About This (Post) by action type - comment"

Data_f.drop('Matched Audience Targeting Consumptions by Type - other clicks', axis = 1, inplace = True)
# we keep "Post Audience Targeting Unique Consumptions by Type - other clicks"
Data_f.drop('Matched Audience Targeting Consumptions by Type - link clicks', axis = 1, inplace = True)
# we keep "Post Audience Targeting Unique Consumptions by Type - link clicks"
Data_f.drop('Matched Audience Targeting Consumptions by Type - video play', axis = 1, inplace = True)
# we keep "Post Audience Targeting Unique Consumptions by Type - video play"

FB_data = Data_f[['Post Total Reach', 'Post organic reach', 'Engaged Users',
                  'Post Message', 'is_paid', 'is_tequila']]
FB_data.to_csv(path_join(output_dir, 'FB_data.csv'), index = False)

if reduce_only:
    sys.exit("Finished data reduction and results saved")


keywords_set_1 = ['electric', 'environment', 'sustainable', 'hybrid', 'co2']
keywords_set_2 = ['mobility', 'share', 'co2', 'safety']
keywords_set_3 = ['city', 'smart', 'co2', 'safety']

# Car enthousiasts
#keywords_set_4 = ['zoe', 'koleos', 'twingo', 'kangoo', 'ez-pro']
keywords_set_4 = ['safe', 'maintenance', 'car service', 'car service', 'car service', ]

#in french it's 'Message de publication'
Data_f['environment'] = Data_f['Post Message'].apply(count_key_message, key_set = keywords_set_1)
Data_f['mobility'] = Data_f['Post Message'].apply(count_key_message, key_set = keywords_set_2)
Data_f['city'] = Data_f['Post Message'].apply(count_key_message, key_set = keywords_set_3)
Data_f['models'] = Data_f['Post Message'].apply(count_key_message, key_set = keywords_set_4)
Data_f.drop('Post Message', axis = 1, inplace = True)

#Data_f['Paid views to 95% eff'] = Data_f['Organic views to 95%'].apply(paid_efficacity, paid = Data_f['Paid views to 95%'])
#Data_f['Paid Video Views eff'] = Data_f['Organic Video Views'].apply(paid_efficacity, paid = Data_f['Paid Video Views'])

Q4 = Data_f[Data_f['Type_Video'] == 1].corr()
Q5 = Data_f[Data_f['Type_Video'] == 0].corr()
Q6 = Data_f.corr()
Q_time = Data_f.corr()

if save_fig:
    plt.subplot(1,3,1)
    sns.heatmap(Q4, cmap="YlGnBu", vmin=-1, vmax=1, xticklabels = True, yticklabels = True)
    
    # we move this one to the right since it's different from the previous ones in terms of dimension
    plt.subplot(1,3,3)
    sns.heatmap(Q5, cmap="YlGnBu", vmin=-1, vmax=1, xticklabels = True, yticklabels = False)
    
    plt.subplot(1,3,2)
    sns.heatmap(Q6, cmap="YlGnBu", vmin=-1, vmax=1, xticklabels = True, yticklabels = False)
    
    plt.savefig(path_join(directory_img, "corr_comparison_after.png"))
    plt.close()
    
    pict_reduced_html = 'reduced_corr.html'
    Html_file = open(path_join(directory_img, pict_reduced_html),"w")
    Html_file.write(Q6.style.background_gradient(cmap='coolwarm').render())
    Html_file.close()

    plt.subplot(1,2,1)
    plt.plot(Data_f['Post Total Reach'].sort_index(), 'k--', linewidth=2)
    #plt.axhline(y = Data_f['Post Total Reach'].mean(), color = 'k')
    plt.plot(Data_f['Post organic reach'].sort_index(), 'r--')
    #plt.axhline(y = Data_f['Post organic reach'].mean(), color = 'r')
    plt.plot(Data_f['Post Paid Reach'].sort_index(), 'g--')
    #plt.axhline(y = Data_f['Post Paid Reach'].mean(), color = 'g')
    plt.title("Facebook Reach")
    plt.legend(("Total", "Organic", "Paid"))
    
    plt.subplot(1,2,2)
    plt.plot(Data_f['Post Total Impressions'].sort_index(), 'k--', linewidth=2)
    #plt.axhline(y = Data_f['Post Total Impressions'].mean(), color = 'k')
    plt.plot(Data_f['Post Organic Impressions'].sort_index(), 'r--')
    #plt.axhline(y = Data_f['Post Organic Impressions'].mean(), color = 'r')
    plt.plot(Data_f['Post Paid Impressions'].sort_index(), 'g--')
    #plt.axhline(y = Data_f['Post Paid Impressions'].mean(), color = 'g')
    plt.title("Facebook Impressions")
    plt.legend(("Total", "Organic", "Paid"))
    
    plt.plot(Data_f['Post Total Reach'].sort_index(), 'k--')
    plt.plot(Data_f['Post organic reach'].sort_index(), 'r')
    
    sns.heatmap(Q_time, cmap="YlGnBu", vmin=-1, vmax=1, xticklabels = True, yticklabels = False)
    
    plt.savefig(path_join(directory_img, "corr_time_blue.png"))
    
    pict_reduced_html = 'corr_time.html'
    Html_file= open(path_join(directory_img, pict_reduced_html),"w")
    Html_file.write(Q_time.style.background_gradient(cmap='coolwarm').render())
    Html_file.close
    
    plt.savefig(path_join(directory_img, "organic_paid.png"), bbox_inches='tight')

W1,p1 = shapiro(Data_f)
Data_no_bool = Data_f.drop(['Type_Video', 'moment_day_Night', 'timestamp', 'is_tequila', 'working_day'], axis = 1)
W2,p2 = shapiro(Data_no_bool)
print("test1 W,p", W1, p1)
print("test2 W,p", W2, p2)

gauss_dict = dict.fromkeys(Data_no_bool.columns)
for col in Data_no_bool.columns:
    _, p_orig = shapiro(Data_f[col])
    col_min = min(Data_f[col])
    if col_min == 0.0:
        col_min = 1e-4
    # boxcox need positive data so we shift everything iff the column has negative value
    trasnformed_data, _ = boxcox(Data_f[col] + col_min)
    _, p_box = shapiro(trasnformed_data)
    gauss_dict[col] = [p_orig, p_box]

print("Total Reach IC no tequila", mean_confidence_interval(
    Data_f[(Data_f['is_tequila'] == 0)]['Post Total Reach'], 0.8))
print("Total Reach IC tequila", mean_confidence_interval(
    Data_f[(Data_f['is_tequila'] == 1)]['Post Total Reach'], 0.8))

Data_video_limited = Data_f[ Data_f['Type_Video'] == 1]
[['Post Total Reach', 'is_tequila', 'timestamp', 'Average time video viewed']]

print('Video only -----------------------------------------------------')

print("Total Reach IC no tequila", mean_confidence_interval(
    Data_video_limited[(Data_video_limited['is_tequila'] == 0)]['Post Total Reach'], 0.8))
print("Total Reach IC tequila", mean_confidence_interval(
    Data_video_limited[(Data_video_limited['is_tequila'] == 1)]['Post Total Reach'], 0.8))
print("Average time video viewed IC no tequila", mean_confidence_interval(
    Data_video_limited[(Data_video_limited['is_tequila'] == 0)]['Average time video viewed'], 0.8))
print("Average time video viewed IC tequila", mean_confidence_interval(
    Data_video_limited[(Data_video_limited['is_tequila'] == 1)]['Average time video viewed'], 0.8))

sns.heatmap(Data_video_limited.corr(), cmap="YlGnBu", vmin=-1, vmax=1, xticklabels = True, yticklabels = False)
print("Total Reach var by night", Data_f[(Data_f['moment_day_Night'] == 1)]['Post Total Reach'].std())
print("Total Reach var by morning", Data_f[(Data_f['moment_day_Night'] == 0)]['Post Total Reach'].std())

print("Total Reach IC by night", mean_confidence_interval(
    Data_f[(Data_f['moment_day_Night'] == 1)]['Post Total Reach'], 0.8))
print("Total Reach IC by morning", mean_confidence_interval(
    Data_f[(Data_f['moment_day_Night'] == 0)]['Post Total Reach'], 0.8))
    
scaler = MinMaxScaler().fit(Data_f)
Data_f_scaled = pd.DataFrame(scaler.transform(Data_f), columns = Data_f.columns)

Data_reduced_scaled = Data_f_scaled.drop(['timestamp', 'moment_day_Night', 'is_tequila', 'working_day', 
                                       'environment', 'mobility', 'city', 'models', 'Type_Video', 'is_paid',
                                       'Organic views to 95%', 'Paid views to 95%', 'Organic Video Views',
                                       'Paid Video Views', 'Average time video viewed', 'Video length',
                                       'Talking About This (Post) by action type - share',
                                       'Talking About This (Post) by action type - like',
                                       'Talking About This (Post) by action type - comment',
                                       'Negative Feedback by Type - hide_all_clicks',
                                       'Negative Feedback by Type - hide_clicks'], axis = 1)

#Data_reduced_scaled = Data_f_scaled.drop(['timestamp', 'moment_day_Night', 'is_tequila', 'working_day', 
#                                       'environment', 'mobility', 'city', 'models', 'Type_Video',
#                                       'Organic views to 95%', 'Paid views to 95%', 'Organic Video Views',
#                                       'Paid Video Views', 'Average time video viewed', 'Video length'], axis = 1)

#Data_reduced_scaled = Data_f_scaled.drop(['timestamp'], axis = 1)

Data_f_scaled_photo = Data_f_scaled[Data_f_scaled['Type_Video'] == 0].drop([
    'timestamp','moment_day_Night', 'is_tequila', 'working_day', 'environment', 'mobility', 'city', 'models', 'Type_Video',
       'Organic views to 95%', 'Paid views to 95%', 'Organic Video Views', 'Paid Video Views', 'Average time video viewed',
       'Video length', 'Talking About This (Post) by action type - share', 'Talking About This (Post) by action type - like',
       'Talking About This (Post) by action type - comment',
       'Negative Feedback by Type - hide_all_clicks', 'Negative Feedback by Type - hide_clicks', 'is_paid'], axis = 1).copy()

Data_f_scaled_video = Data_f_scaled[Data_f_scaled['Type_Video'] == 1].drop([
    'timestamp', 'moment_day_Night', 'is_tequila', 'working_day', 'environment', 'mobility', 'city', 'models', 'Type_Video',
      'Talking About This (Post) by action type - share', 'Talking About This (Post) by action type - like',
      'Talking About This (Post) by action type - comment',
      'Negative Feedback by Type - hide_all_clicks', 'Negative Feedback by Type - hide_clicks', 'is_paid'], axis = 1).copy()

precision = 0.90

pca_obj = PCA(precision)
pc_data = pca_obj.fit_transform(Data_reduced_scaled)
print(pca_obj.explained_variance_ratio_)

pca_load = pca_obj.components_
print(pca_load.shape)
if save_fig:
    sns.heatmap(pca_load, annot = True, center = 0, xticklabels = Data_reduced_scaled.columns, yticklabels = False)
    
pca_obj_photo = PCA(precision)
pc_data_photo = pca_obj_photo.fit_transform(Data_f_scaled_photo)
print(pca_obj_photo.explained_variance_ratio_)

pca_load_photo = pca_obj_photo.components_
print(pca_load_photo.shape)

if save_fig:
    sns.heatmap(pca_load_photo, annot = True, center = 0, xticklabels = Data_f_scaled_photo.columns, yticklabels = False)
    
#We in order to use ANOVA we must have all the labels without spaces otherwise formulas won't work
Data_ANOVA = Data_f_scaled.copy()
new_list = []
for i in range(len(Data_ANOVA.columns)):
    new_list.append(Data_ANOVA.columns[i].replace(" ", "_"))

Data_ANOVA.columns = new_list

Reach_ANOVA = Iterative_ANOVA(Data_ANOVA, 'Post_Total_Reach',
                              #['Type_Video', 'is_tequila', 'is_paid', 'working_day', 'moment_day_Night'], 0.5)
                              ['Type_Video', 'is_tequila', 'is_paid', 'working_day', 'moment_day_Night'], 0.01)
Reach_model = Reach_ANOVA.fit()
Reach_rel_cat = Reach_ANOVA.get_curr_list()

sm.stats.anova_lm(Reach_ANOVA.history[3], typ=2)

"""
Impression_ANOVA = Iterative_ANOVA(Data_ANOVA, 'Post_Total_Impressions',
                              ['Type_Video', 'is_tequila', 'is_paid', 'working_day', 'moment_day_Night'], 0.01)
Impression_model = Impression_ANOVA.fit()
Impression_rel_cat = Impression_ANOVA.get_curr_list()
"""

min_clust = 2
max_clust = 10
n_clust_vec = range(min_clust, max_clust)
metric_str = 'euclidean'
# inertie silhouette 

# kmeans works well with n-dim balls
k_means_list = [];
k_means_results = [];
k_means_silhouettes = [];

fig = plt.figure(figsize=(20, 80))
for i in n_clust_vec:
    k_means_list.append(KMeans(n_clusters = i, random_state=0))
    k_means_results.append(k_means_list[i-min_clust].fit_predict(Data_reduced_scaled))
    k_means_silhouettes.append(silhouette_score(Data_reduced_scaled, k_means_results[i-min_clust], metric = metric_str))
    ax = fig.add_subplot(max_clust-min_clust, 1, i -min_clust + 1, projection='3d')
    ax.scatter(pc_data[:,0], pc_data[:,1], pc_data[:,2], c = k_means_results[i-min_clust], cmap='viridis', marker='o')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    fig.savefig(path_join(directory_img,"k_m_cluster_{}.png".format(i)))
    plt.close()
    
print(k_means_silhouettes)

# DBSCAN works well with more complex shapes based on density
# find avg distance inter and intra cluster
DBSCAN_0 = DBSCAN().fit_predict(Data_f_scaled)
DBSCAN_1 = DBSCAN().fit_predict(Data_f_scaled)
DBSCAN_2 = DBSCAN().fit_predict(Data_f_scaled)

# SpectralClustering works well with more complex shapes based on density
SpectralClustering_0 = SpectralClustering().fit_predict(Data_f_scaled)
SpectralClustering_1 = SpectralClustering().fit_predict(Data_f_scaled)
SpectralClustering_2 = SpectralClustering().fit_predict(Data_f_scaled)

score_lst = []
"""
score_lst.append(silhouette_score(Data_f_scaled, kmeans_0, metric = metric_str) )
score_lst.append(silhouette_score(Data_f_scaled, kmeans_1, metric = metric_str) )
score_lst.append(silhouette_score(Data_f_scaled, kmeans_2, metric = metric_str) )
"""
score_lst.append(silhouette_score(Data_f_scaled, DBSCAN_0, metric = metric_str) )
score_lst.append(silhouette_score(Data_f_scaled, DBSCAN_1, metric = metric_str) )
score_lst.append(silhouette_score(Data_f_scaled, DBSCAN_2, metric = metric_str) )
score_lst.append(silhouette_score(Data_f_scaled, SpectralClustering_0, metric = metric_str) )
score_lst.append(silhouette_score(Data_f_scaled, SpectralClustering_1, metric = metric_str) )
score_lst.append(silhouette_score(Data_f_scaled, SpectralClustering_2, metric = metric_str) )

print(score_lst)

for i in n_clust_vec:
    fig = plt.figure(figsize=(20, 40))
    model = SilhouetteVisualizer(KMeans(i, random_state=0))
    model.fit(Data_reduced_scaled)
    model.show()    
    fig.savefig(path_join(directory_img, "k_m_silhouettes_{}.png".format(i) ))
    plt.close()

Data_TSNE_2 = TSNE(n_components=2).fit_transform(Data_reduced_scaled)
Data_TSNE_3 = TSNE(n_components=3).fit_transform(Data_reduced_scaled)

fig = plt.figure(figsize=(20, 40))
Data_Video = Data_f[Data_f['Type_Video'] == 1]
ax = fig.add_subplot(2, 1, 1)
ax.scatter(Data_TSNE_3[:,0], Data_TSNE_3[:,1], c = k_means_results[0], cmap='viridis', marker='o', s = 30)
ax.set_xlabel('TSNE_2 1')
ax.set_ylabel('TSNE_2 2')

ax = fig.add_subplot(2, 1, 2, projection='3d')
ax.scatter(Data_TSNE_3[:,0], Data_TSNE_3[:,1], Data_TSNE_3[:,2], c = k_means_results[0], cmap='viridis', marker='o', s = 30)
ax.set_xlabel('TSNE_3 1')
ax.set_ylabel('TSNE_3 2')
ax.set_zlabel('TSNE_3 3')

fig.savefig(path_join(directory_img, "tsne.png"))
