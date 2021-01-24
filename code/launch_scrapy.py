# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:46:31 2020

Script used to launch scrapy from the IPython console (version for Spyder).

Attention: Scrapy cannot be launched several times in the same console, it is
therefore necessary to launch different consoles for different launches

@author: lfiorentini
version: 1.1
"""

import os
from global_var_nlp import project_list, dir_from_project
from os.path import join as path_join  # import system for path definition
from spider_results import spider_results
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help = "file to be parsed", type = str,
                        default = 'ces2020')
    parser.add_argument("-p", help = "project", type = str,
                        default = 'renault', choices = project_list)
    site = str(parser.parse_args().f).replace("\n", "")
    project = str(parser.parse_args().p).replace("\n", "")
    
    input_dir, output_dir = dir_from_project(project)
#    print(project, input_dir, output_dir)
    encoding = 'utf8'
    passing_file = path_join(input_dir, 'site_to_scrap.txt')
    with open(passing_file, 'w', encoding=encoding) as filehandle:
        filehandle.write('%s' % site)
        filehandle.close()
    out_file = path_join(output_dir, site, 'store.json')
    #remove existing file in order to avoid appending at the end of it
    try:
        os.remove(out_file)
    except FileNotFoundError:
        pass
#        print('nothing to remove')
    item_list = spider_results(site, project)
