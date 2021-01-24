# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 00:00:00 2020

Top level module to perform keywords search per iteration.
In the first step it triggers google search to obtain input files.
Then it launches scrapy spider per each input file followed by pipeline
processing. It counts the top keywords derived from the websites scraped in
current iteration. Those freshly obtained keywords are passed to start another
iteration. The process continues until given number of iterations is reached
or all possible search terms has been used within whole process.

@author: lfiorentini, mravaine, asys, esavini
version 1.5
"""

from global_var_nlp import dir_from_project, list_from_json
import os
import argparse
import time
from iterative_functions import keywords_per_site, keywords_count_per_iteration

parser = argparse.ArgumentParser()
parser.add_argument("-i", help = "number of iterations", type = int,
                        default = 2)
parser.add_argument("-l", help = "language", type = str,
                        default = "en")
max_it = int(parser.parse_args().i)
language = str(parser.parse_args().l)
start_time = time.time()
if __name__ == "__main__":
    for iteration in range(max_it):
        iteration_start = time.time()
        print('iteration %s in progress' % iteration)
        os.system('python iterative_search.py')
        print('google search done')
        site_list = list_from_json('../iterative', 'site_list.json',
                              'site_list')
        site_list_str = ' '.join(site_list)
        print('site_list: ' + site_list_str)
        for el in site_list:
            passing_file = '../input/site_to_scrap.txt'
            with open(passing_file, 'w', encoding = 'utf8') as filehandle:
                filehandle.write('%s' % el)
                filehandle.close()
            os.system('python launch_scrapy.py -p iterative -f ' + el)
        os.system('python scraper_pipeline.py -p iterative -f ' + site_list_str)
        # Calculate keywords per site.
        keywords_per_site(site_list)
        # Count which keywords appear most frequently between the sites
        # within one iteration.
        keywords_count_per_iteration()
        print("Iteration %s time: %s seconds "
              % (iteration, round((time.time() - iteration_start), 2)))
    print('Process consisting of %s iterations took: %s seconds' % (max_it,
          round((time.time() - start_time), 2)))
