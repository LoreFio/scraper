# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:46:12 2020

@author: lfiorentini
version: 1.1
"""
from global_var_nlp import dir_from_project, project_list
from semantic_search import Semantic_search
import time
import argparse

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("-f", help="list of sites", nargs="*",
                        default=[])
parser.add_argument("-w", help="list of words", nargs="*",
                        default=[])
parser.add_argument("-p", help = "project", type = str,
                        default = 'renault', choices = project_list)

site_list = list(parser.parse_args().f)
w_list = list(parser.parse_args().w)
project = str(parser.parse_args().p).replace("\n", "")

"""
INITIALISATION

"""

percentage_n = 0.01
k_cluster = 2

_, output_dir, = dir_from_project()
for word in w_list:
    print('word:', word)
    for site in site_list:
        print('site:', site)
        # quantiles are printed from semantic_search.py
        sem_src = Semantic_search(output_dir, site, percentage = percentage_n,
                                  language = "english", normalizer = "spacy")
        sem_src.is_keyword(word)
        syn_used, similar = sem_src.sem_sphere(word)
        syn_unique = []
        # skip printing the keyword in synonyms
        for syn_used in [x for x in syn_used if x != word]:
            syn_unique.append(syn_used)
        # skip printing empty synonyms
        if syn_unique != []:
            print('used synonyms', sorted(syn_unique))
        clean = [i[0] for i in similar]
        # skip printing empty similar words
        if clean != []:
            print('similar words', sorted(clean))
# The next line is a basic clustering algorithm for the words
#    KM_model, center_closest, score, words = sem_src.clustering(k_cluster)
final_time = time.time()
print("Total time %s seconds" % (final_time - start_time))
