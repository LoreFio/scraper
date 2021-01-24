# -*- coding: utf-8 -*-
"""
Created on Wed Jun 03 16:50:00 2020

This module exposes certain functionalities to end user. It gives him
the freedom to compare and analyse the keyword results as she/he wants.

It also allows to propose new keywords and domains to scrape, while checking
if they were not used already.

We expect keywords to repeat while executing the iterative process.
We are building a snapshot of the car manufacturers market
reflected in keywords and domains database.
The space (intersection) where the domains and audiences overlap is the actual
body of the market.

@author: asys
version 1.1.0.
"""

import os
import sys
from global_var_nlp import dir_from_project, n_keyword, useless_w_list,\
    path_join
from keyword_finder import Keyword_finder
from collections import Counter
import json


class SteeringWheel:
    def __init__(self):
        pass
    def compare_domains(self):
        """
        Compare if domains have similar content.

        Returns :

        joint_keywords for two domains
        joint_ratio

        """
        project = 'renault'
        _, output_dir = dir_from_project(project)

        selected_domains = ['valeo', 'whatcar']

        percentage_key = 0.25
        col_min = 1

        finder = Keyword_finder(selected_domains, output_dir, n_keyword=n_keyword,
                                useless_w_list=useless_w_list)

        all_kwords = finder.all_keyword()
        joint_kwords, support_dict = finder.common_keyword(col_min, percentage_key)
        joint_ratio = (round(len(joint_kwords) / len(all_kwords) * 100, 2))
        print('Joint keywords for %s and %s are:' % (selected_domains[0], selected_domains[1]), sorted(joint_kwords))
        # printing the ratio to judge the model configuration
        print('joint_ratio:', joint_ratio, '%')


    def top_keyword_per_audience(self):
        """
        Returns :

        top keywords per given audience sorted by number of appearances
        """
        selected_audience = 'Car enthusiasts'
        input_dir = '../kwords_db'
        all_key = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)) and 'json' in f:
                with open(path_join(input_dir, f), 'r') as json_file:
                    data = json.load(json_file)
                    for key in data:
                        if key == 'joint_kwords' and \
                                data['audience'] == selected_audience:
                            site_kw = [el for el in data[key]]
                            all_key = sorted(all_key + site_kw)

        count = Counter(word for word in all_key)
        print('Top %s keywords for %s with number of appearances:'
              % (len(all_key), selected_audience), '\n')
        for value, count in count.most_common():
            print(value, count)


    def compare_audiences(self):
        """
        Compare if audiences have similar content.

        Returns :

        joint_keywords for two audiences
        joint_ratio

        """
        project = 'renault'
        _, output_dir = dir_from_project(project)
        selected_audiences = 'Car enthusiasts'
        input_dir = '../kwords_db'
        sys.path.append(input_dir)
        sub_site = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)) and 'json' in f:
                with open(path_join(input_dir, f), 'r') as json_file:
                    data = json.load(json_file)
                    if data['audience'] == selected_audiences:
                        for key in data:
                            name = data['name']
                            if name not in sub_site:
                                sub_site.append(name)

        print(sub_site)
        percentage_key = 0.25
        col_min = 1

        finder = Keyword_finder(sub_site, output_dir, n_keyword=n_keyword,
                                useless_w_list=useless_w_list)

        all_kwords = finder.all_keyword()
        joint_kwords, support_dict = finder.common_keyword(col_min, percentage_key)
        joint_ratio = (round(len(joint_kwords) / len(all_kwords) * 100, 2))
        print('joint =', sorted(joint_kwords))
        # printing the ratio to judge the model configuration
        print('joint_ratio:', joint_ratio, '%')


    def sites_homogenity(self):
        """
        Function to determine, whether a set of given domains
        is homogeneous or not. The answer is based on joint_kwords ratio
        between the websites.
        """
        set_of_domains = []
        joint_kwords = []


    def kwords_from_sem_sphere(self):
        """
        Count the words appearance in semantic sphere for all keywords
        within one domain. Order them by highest number of appearances.
        Choose new keywords from them.
        """


    def kword_vs_sem_sphere_intersection(self):
        """
        Check the overlapping between given keyword and semantic sphere
        (synonym, similar word) from another website.
        Pass results from key_sem.py.
        """
        kwords_site_a = []
        synonym_site_b = []
        similar_words_site_b = []
        kwords_a_in_syn_b = []
        kwords_a_in_sim_b = []
        for kword in kwords_site_a:
            if kword == synonym_site_b:
                kwords_a_in_syn_b.append(kword)
            elif kword == similar_words_site_b:
                kwords_a_in_sim_b.append(kword)
            else:
                pass


    def get_kword_from_user(self):
        """
        Take input from end user and check, if proposed combination of keywords
        was already used. Reuse kword_already_used function from keyword_finder.
        """
        new_kwords_proposal = []


    def get_domains_from_user(self):
        """
        Take input from end user and check, if proposed combination of domains
        was already used. Reuse domain_already_scraped function from keyword_finder.
        """
        new_domains_proposal = []


    def kword_match_for_audience(self):
        """
        Check if set of proposed keywords is a good match to create content
        for a given audience.
        """
