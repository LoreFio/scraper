# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:59:30 2020

@author: lfiorentini, asys
version 1.2
"""

from global_var_nlp import smartirs_list, dir_from_project, sub_dict,\
    project_list, list_to_json
from os.path import join as path_join  # import system for path definition
from site_processer import Site_processer
import time
import json
import argparse
import logging
from datetime import datetime

start_time = time.time()

"""
INITIALISATION

"""

_, output_dir = dir_from_project()
any_num = True
percentage = 0.1

"""
PROCESSING SITE BY SITE

"""

time_stamp = "{:%Y_%m_%d_%Hh_%Mm}".format(datetime.now())
logging.basicConfig(filename='../logs/%s_iterative_pipeline.txt' % time_stamp,
                    level='INFO')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help = "project", type = str,
                        default = 'renault', choices = project_list)
    parser.add_argument("-f", help="list of sites", nargs = "*",
                        default = ['ces2020'])
    project = str(parser.parse_args().p).replace("\n", "")
    site_list = list(parser.parse_args().f)
    
    input_dir, _ = dir_from_project(project)

    if len(site_list) == 0:
        with open(path_join(input_dir, 'site_list.json'), 'r',
                  encoding = 'utf8') as json_file:
            data = json.load(json_file)
            json_file.close()

        site_list = data['site_list']
    for site in site_list:
        site_start_time = time.time()
        site_proc = Site_processer(site, output_dir, sub_dict,
                                   normalizer = "spacy", language = None)
        site_proc.open_n_divide()
        site_norm_list = site_proc.load_n_normalize(any_num)
        site_content = site_proc.create_table(site_norm_list, smartirs_list,
                                                  percentage)
        site_proc.save_table(site_content)
        print(site, site_proc.__language__,
              "dataframe created in %s seconds ---" %
              round((time.time() - site_start_time), 2))
    
    print("Creating dataframes total time: %s seconds "
          % round((time.time() - start_time), 2))
