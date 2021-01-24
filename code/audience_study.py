# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:46:12 2020

@author: lfiorentini, asys
version: 1.2
"""
from keyword_finder import Keyword_finder
from global_var_nlp import n_keyword, dir_from_project, percentage_key,\
    useless_w_list, col_min, get_site_list, list_to_json, project_list
import time
import json
import argparse

start_time = time.time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help = "project", type = str,
                        default = 'renault', choices = project_list)
    project = str(parser.parse_args().p).replace("\n", "")
    
    input_dir, output_dir = dir_from_project(project)
    site_list = get_site_list(project)
    kwords_time = time.time()
    
    # reading audience assignement from input_description.csv is deactivated.
    # in next steps we want to read it from kw.json files
    
    if len(site_list) > 0:
        # we work only on the not empty lists
        finder = Keyword_finder(site_list, output_dir, n_keyword = n_keyword,
                                useless_w_list = useless_w_list)
    
        all_kwords = finder.all_keyword()
        joint_kwords, support_dict = finder.common_keyword(col_min, percentage_key)
        joint_ratio = (round(len(joint_kwords) / len(all_kwords) *100, 2))
        
        '''
        Saving results to json.
        '''
        file = '../kwords_db/kw.json' 
        data = {}
        data['language'] = ''  # pass from lang detect
        data['keyword_list'] = '%s' % sorted(joint_kwords[0:4])
        data['site_list'] = '%s' % site_list
        data['audience'] = ''
        data['business domain'] = None
        # implement logic to recognize business domain from scrape results
        data['media channel used'] = None
        # implement logic to recognize media channel used from scrape results
        data['len_all_kwords'] = len(all_kwords)
        data['n_keyword'] = n_keyword
        data['col_min'] = col_min
        data['joint_ratio'] = joint_ratio
        data['joint_kwords'] = '%s' % sorted(str(el) for el in joint_kwords)
        data['all_kwords'] = sorted(str(el) for el in all_kwords)
        
        with open(file, 'w') as outfile:
            json.dump(data, outfile, indent = 4, sort_keys = False)
            outfile.close()
    
        list_to_json(output_dir, 'keyword_list.json', 'keyword_list',
                     joint_kwords[0:4])
    
    print("\nCreating joint_kwords total time: %s seconds "
              % round((time.time() - kwords_time), 2))
