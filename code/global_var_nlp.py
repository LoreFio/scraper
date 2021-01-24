# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:27:25 2020

@author: lfiorentini
"""
from os.path import join as path_join  # import system for path definition
import pandas as pd
import os
import json
import re
import argparse


project_list = ['renault', 'delifrance', 'iterative']

useless_site_list_deli = ['panistar.com.csv']

site_list_deli = [re.sub(r'.csv','', file) for file in
                  os.listdir('../input_delifrance/')
                  if file not in useless_site_list_deli]

useless_site_list_renault = ['input_description.csv', 'site_to_scrap.txt',
                             'prototype.json']

site_list_renault = [re.sub(r'.json','', file) for file in os.listdir('../input/')
             if file not in useless_site_list_renault]
site_list_renault.sort()

languages_list = ['english', 'french']

sub_dict = {r'\\xa0': " ",                      # white space
            r'\\n': "",                         # remove \n
            r'\\r': "",                         # remove \r
            r'\\t': "",                         # remove \t
            r'\xa0': " ",                      # white space
            r'\n': "",                         # remove \n
            r'\r': "",                         # remove \r
            r'\t': "",                         # remove \t
            r'&amp': "",                        # remove &
            r'[\"\'\u2018\u2019]': " ",         # remove quotes
            r'www\.[^ ]*': "",                   # remove links
            r'[^ ]*@[^ ]*': "",                 # remove mail addresses
            r'[^ ]*_[^ ]*': "",                 # remove words with _
            r'data.offers[^ ]*': "",           
            # remove data.offers needed by lesnumeriques_fr
            r'([\U00002600-\U000027BF])|\
                ([\U00010000-\U0010ffff])': "", # emojis and others
            r'[Oo][Nn][’\"]*[Aa][Ii][Rr]':
                "onair",                        # transform the onair
            r'[^\w\s\'\’]': " ",                 # punctuations
            r' +': " ",                         # multiple spaces
            r'^\s': "",                         # spaces at the beginning
            }
    
smartirs_list = ['nnc', 'nnn', 'anc']
'''
the full smartirs_list initially considered was 
smartirs_list = ['nfc', 'nnc', 'nfn', 'nnn', 'lnc', 'lnn', 'ann', 'afn', 'afc',
'anc']

the following couples are highly correlated and will be therefore not
considered twice:
    (nfc, lnc, nnc, ann) => nnc
    (nfn, nnn, lnn) => nnn

all the l** with the correspective n**
afn won't be considered because it involves a very negative correlation with
the frequency in the header and all correlations with freaquencies in the
sections are negative.
afc won't be considered since it has provided poor results on both ces2020 and
renault_onair
'''

useless_w_list = ['subscribe', 'email', 'isnt', 'instagram', 'twitter',
                  'facebook', 'youtube', 'linkedin', 'follow', 'would', 'also',
                  'aussi']

audiences = ['Tech pioneers', 'Car enthusiasts', 'Conscious thinkers',
             'City makers']

percentage_key = 0.25
n_keyword = 40
col_min = 3

meta_domains = ['play_google', 'apps_apple', 'facebook', 'alibaba', 'amazon']

car_manuf_keyword = ["audi", "auto", "automotive", "autonomous", "battery",
                     "bmw", "buy", "car", "charge", "chevrolet", "city",
                     "design", "diesel", "drive", "electric", "energy", "ev",
                     "find", "ford", "fuel", "future", "good", "honda",
                     "hybrid", "insurance", "intermodality", "jeep", "kia",
                     "mercedes", "mobility", "model", "new", "nissan", "offer",
                     "power", "prius", "range", "rear", "renault", "safety",
                     "sharing", "smart", "suv", "system", "tech", "tesla",
                     "toyota", "vehicle", "zoe"]

http_protocols = ['http:\/\/', 'https:\/\/', 'www.']

def list_from_json(input_dir = '../iterative', file = 'keyword_list.json',
                   field_name = 'keyword_list'):
    """
    This function retrieves a list of objects saved in a json file

    Parameters
    ----------
    input_dir : str, optional
        Directory where we are going to operate. The default is './iterative'.
    file : str, optional
        Filename. The default is 'keyword_list.json'.
    field_name : str, optional
        Name of the field of the dictionary. The default is 'keyword_list'.

    Returns
    -------
    list
        list of objects.

    """
    try :
        with open(path_join(input_dir, file), 'r',
                  encoding = 'utf8') as json_file:
            data = json.load(json_file)
        json_file.close()
        return data[field_name]
    except FileNotFoundError:
        return []        

def list_to_json(input_dir = '../iterative', file = 'keyword_list.json',
                   field_name = 'keyword_list', list_object = []):
    """
    This function saves a list of objects saved in a json file

    Parameters
    ----------
    input_dir : str, optional
        Directory where we are going to operate. The default is './iterative'.
    file : str, optional
        Filename. The default is 'keyword_list.json'.
    field_name : str, optional
        Name of the field of the dictionary. The default is 'keyword_list'.
    list_object : list, optional
        objects to be saved
    Returns
    -------
    list
        list of objects.

    """
    data = dict()
    data[field_name] = list_object
    with open(path_join(input_dir, file),
              'w', encoding='utf8') as json_file:
        json.dump(data, json_file, indent = 4)
        json_file.close()
        
def get_site_list(project ='iterative'):
    """
    This function reads the argument passed to the script defining the list of
    sites to work on. If nothing is passed as argument it will read the file
    'site_list.json'
    Returns : 
    
    site_list  : list
        list of sites to work on.

    """
    input_dir, _ = dir_from_project(project)
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="list of sites", nargs="*",
                        default=[])
    site_list = list(parser.parse_args().f)
    if len(site_list) == 0:
        with open(path_join(input_dir, 'site_list.json'), 'r',
                  encoding = 'utf8') as json_file:
            data = json.load(json_file)
            json_file.close()

        site_list = data['site_list']
    return site_list

def dir_from_project(project = 'renault'):
    """
    This function return the project input directory and output one.

    Parameters
    
    project : str, optional
        project name. The default is 'renault'.

    Returns
    
    in_dir : str
        input directory of the project.
    out_dir : str
        output directory of the project.

    """
    if project == 'renault':
        in_dir = '../input'
        out_dir = '../output'
    elif project == 'delifrance':
        in_dir = '../input_delifrance'
        out_dir = '../output_delifrance'
    elif project == 'iterative':
        in_dir = '../iterative'
        out_dir = '../output'
    return in_dir, out_dir

def get_description(project = 'renault'):
    """
    This function return the project description files

    Parameters
    
    project : str, optional
        project name. The default is 'renault'.

    Returns
    -------
    description : pd.DataFrame
        input_description file read properly.

    """
    input_dir, _ = dir_from_project(project)
    description = pd.read_csv(path_join(input_dir, 'input_description.csv'),
                          sep = ';')

    description.fillna("", inplace = True)

    return description

def renault_lists():
    """
    This function define the principale lists needed by the
    code for the Renault project

    Returns
    -------
    brand_list : list
        brands list inside the input description file.
    renault_list : list
        renautl sites in the input description file.

    """
    description = get_description(project = 'renault')    
    brand_list = list(
        description[description['Domain'] == 'Car brand']['Description'])
    
    renault_list = list(description[description['Description'].str.startswith(
        'renault')]['Description'])
    
    return brand_list, renault_list
