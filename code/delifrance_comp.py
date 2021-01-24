# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:19:17 2020

@author: asys
@author: lfiorentini
version 1.1
"""

from os.path import join as path_join# import system for path definition
from global_var_nlp import dir_from_project, project_list, site_list_deli
from deli_utilities import find_opportunities, contain_abbreviations
import pandas as pd
import time
import argparse

def compare_table(criterium = 0):
    """
    This function look for concurrents' keywords according to a criterium and
    return a table with all the concurrents' performances as well as the client
    ones

    Parameters :
    
    criterium : int, optional
        Type of the criterium. The default is 0.

    Returns :
    
    res : pd.DataFrame
        DataFrame with the resulting table.

    """
    res = pd.DataFrame(columns = new_cols)
    for site in site_list_deli:
        if site != client_site:
            if criterium == 0:
                kw_list = top_dict[site]
            elif criterium == 1:
                kw_list = data_dict[site][
                    data_dict[site]['Position'] < 10]['Keyword']
            for kw in kw_list:
                if not contain_abbreviations(kw, site_abbreviations):
                    line = data_dict[site][data_dict[site]['Keyword'] == kw]
                    line.insert(len(line.columns), extra_col[0], site)
                    client_row = data_dict[
                        client_site][data_dict[client_site]['Keyword'] == kw]
                    if len(client_row) != 0:
                        client_traffic = sum(client_row['Traffic'])
                        client_pos = min(client_row['Position'])
                    else:
                        client_traffic = 0
                        client_pos = no_position
                    line.insert(len(line.columns), extra_col[1],
                                client_traffic) 
                    line.insert(len(line.columns), extra_col[2], client_pos) 
                    res = pd.concat([res, line], ignore_index = True)
    return res

parser = argparse.ArgumentParser()
parser.add_argument("-p", help = "project", type = str,
                    default = 'renault', choices = project_list)
project = str(parser.parse_args().p).replace("\n", "")
project = 'delifrance'        
start_time = time.time()

"""
INITIALISATION

"""

top_n = 100
bad_pos = 9
q = 0.75
no_position = 1000
input_dir, output_dir = dir_from_project(project)
data_dict = {}
top_dict = {}
opportunities_dict = {}
client_site = 'delifrance.com'
col_interest = 'Traffic'

for site in site_list_deli:
    data_dict[site] = pd.read_csv(path_join(input_dir, site + '.csv'),
                                  sep = ',')
    data_dict[site].drop(['CPC', 'URL', 'Previous position', 'Traffic Cost',
                          'Trends', 'Timestamp', 'SERP Features by Keyword'],
                         axis = 1, inplace = True)
    print(site, len(data_dict[site]))
    top_dict[site] = set(data_dict[site].sort_values(
        col_interest, ascending = False)['Keyword'].head(top_n))
    
    if site == site_list_deli[0]:
        common = set(data_dict[site]['Keyword'])
    else:
        common.intersection_update(set(data_dict[site]['Keyword']))
    
    opportunities_dict[site] = find_opportunities(data_dict[site],
                                                  bad_pos = bad_pos,
                                                  q = q)

"""
sugg_kword 

"""

common_client = pd.DataFrame(columns = ['word', 'competitor',
                                        'competitor traffic',
                                        'competitor position', 'deli traffic',
                                        'deli position'])

for kword in common:
    line = [kword, '', 0, 0, 0, 0]
    for site in site_list_deli:
        site_row = data_dict[site][data_dict[site]['Keyword'] == kword]
        site_traffic = sum(site_row['Traffic'])
        site_pos = min(site_row['Position'])
        if site == client_site:
            line[1] = site
            line[4] = site_traffic
            line[5] = site_pos
        elif site_traffic > line[2]:
            line[1] = site
            line[2] = site_traffic
            line[3] = site_pos
    common_client = pd.concat([common_client, line], ignore_index = True)

"""
sugg_kword 

"""

extra_col = ['competitor', 'deli traffic', 'deli position']
new_cols = list(data_dict[client_site].columns).append(extra_col)
site_abbreviations = ['aryzta', 'arytza', 'bridor', 'coup de p', 'coupe pate',
                      'europastry', 'lantmannen', 'lantmännen', 'unibake',
                      'neuhauser', 'panistar', 'transgourmet', 'vandemoortele']


sugg_kword = compare_table(criterium = 0)
sugg_kword.sort_values('Position', ascending = True, inplace = True)
sugg_kword.drop_duplicates(subset = ['Keyword', extra_col[0]],
                           keep='first', inplace = True)

"""
position_kword 

"""

position_kword = compare_table(criterium = 1)

"""
SAVE DATA

"""

writer = pd.ExcelWriter(path_join(output_dir, 'deli_result.xlsx'),
                        engine='xlsxwriter')

# Write each dataframe to a different worksheet.
sugg_kword.to_excel(writer, sheet_name = 'suggested_keywords', index = False)
common_client.to_excel(writer, sheet_name = 'common_keywords', index = False)
position_kword.to_excel(writer, sheet_name = 'position_keywords', index = False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()
