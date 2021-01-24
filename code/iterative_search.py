# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:06:07 2020

This module implements google search into scraping framework.
It allows to get URLs straight from google and pass them to spider
and further processing.

@author: lfiorentini, asys, esavini
version 1.6.0
"""

import re
from itertools import combinations, product
from global_var_nlp import list_to_json, list_from_json, meta_domains, \
    http_protocols
from iterative_functions import keyword_already_used, site_already_scraped, \
    single_google_search
from save2json import Save2Json


if __name__ == "__main__":
    """
    Perform a google search for given list of keyword combinations.
    
    Parameters :
    
    must have keys : list
        One of these keywords is included in every search query.
    keys : list
        Initial list of keywords from which the combinations are created.
    term : list
        Single combination of keywords passed to search query.
    site_url : url
        One of URLs given in the results page of google search.
    http_cut : 
        Variable used to subtract http protocols and other content of URLs
        by using regex. In order to elicit top level domain and name per site.
    
    Returns :
    
    input_file : json
        Contains name, top level domain and site url.
    site_list : list
        List of input files to scrape.    
    
    """

    site = ''
    site_list = []
    urls_list = []
    must_have_keys = ['car', 'renault']
    m_h_comb = combinations(must_have_keys, 1)
    keyword_list = list_from_json('../iterative', 'keyword_list.json',
                              'keyword_list')
    # Choose how many keywords should pass from top of current iteration to
    # begin search for next iteration.
    keys = keyword_list[0:5]
    for el in must_have_keys:
        if el in keys:
            keys.remove(el)
    keys_comb = combinations(keys, 2)
    # Create all possible three word combination of one of the must have
    # keywords and given keyword list.
    for el in product(m_h_comb, keys_comb):
        i = list(sum(el, ()))
        if keyword_already_used(i) == False:
            term = ' '.join(i)
            print('search keywords: ', term)
            for site_url in single_google_search(term, num_results=10,
                                                 lang="en", lr="lang_en",
                                                 cr="countryFR", gl="fr"):
                if '.pdf' not in site_url:
                    # Remove http protocols from urls.
                    http_cut = site_url
                    for el in http_protocols:
                        http_cut = re.sub(r'^%s' % el, '', http_cut)
                    # Get top level domain eg. 'renault.com'
                    top_level_domain = re.sub(r'/(.*)', '', http_cut)
                    if http_cut.count('.') > 1:
                        http_cut = re.sub(r'^[\w]*\.', '', http_cut)
                    # Get site name eg. 'renault'
                    site = re.sub(r'\..+$', '', http_cut)
                    if site not in meta_domains:
                        if site not in site_list:
                            if "/" not in site:
                                # above condition allows iterative_process flow
                                # but we loose urls. If regex can be improved,
                                # release this condition
                                i = site_already_scraped(site)
                                if i is not None:
                                    urls_list.append(site_url)
                                    # Save above elements in input file used
                                    # in scraping process.
                                    input_file = Save2Json(i, site_url,
                                                           top_level_domain)
                                    site_list.append(i)
    for el in sorted(urls_list):
        print(el)
    list_to_json('../iterative', 'site_list.json', 'site_list',
                 site_list)
    print('%s new websites found' % len(site_list))
