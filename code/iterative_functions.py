# -*- coding: utf-8 -*-
"""
Created on Tue Sep 08 15:36:00 2020

Module containing functions used in iterative process.

@author: asys
version 1.1.0
"""

from global_var_nlp import list_to_json, list_from_json, dir_from_project
from requests import get
from bs4 import BeautifulSoup
from site_keyword import Site_keyword
from os.path import join as path_join
import json
import os
from collections import Counter


def keyword_already_used(el):
    """'
    Check if the keywords nominated to be used in following iteration
    has not been used previously.

    Parameters :

    used_keyword_list : dictionary
        A json dictionary of keyword combinations already used in previous
        searches.
    el : list
        A single combination of keywords.

    Returns :

    False : if combination was not used yet.
    True : if combination was used already.

    """
    used_keyword_list = list_from_json('../iterative',
                                       'used_keyword_list.json',
                                       'used_keyword_list')
    sets = [set(i) for i in used_keyword_list]
    if set(el) not in sets:
        used_keyword_list.append(el)
        list_to_json('../iterative', 'used_keyword_list.json',
                     'used_keyword_list', used_keyword_list)
        return False
    else:
        print('set already used:', el)
        return True


def site_already_scraped(el):
    """'
    Check if domain nominated for scrape has not been used previously.

    Parameters :

    scraped_site_list : dictionary
        A json dictionary of sites already scraped in previous iterations.

    Returns :

    el : str
        A site name that was not scraped yet.
    """
    scraped_site_list = list_from_json('../iterative',
                                       'scraped_site_list.json',
                                       'scraped_site_list')
    if el not in scraped_site_list:
        scraped_site_list.append(el)
        list_to_json('../iterative', 'scraped_site_list.json',
                     'scraped_site_list', scraped_site_list)
        return el


def single_google_search(term, num_results, lang, lr, cr, gl):
    """
    Modification of package: googlesearch-python v2020.0.2.
    Perform google search for keywords in specified language.

    Parameters :

    term : list
        List of words to search.
    num_results : int
        Number of urls returned.
    lang : str
        User interface language (host language).
    lr : str
        Language of the documents received.
    cr : str
        Restrict search results to documents originating in a particular
        country.
    gl : str
        Boost search results for country of origin.

    Returns :

    list of urls

    """
    usr_agent = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/61.0.3163.100 Safari/537.36'}

    def fetch_results(search_term, number_results, language_code):
        """
        Get the query results from google search.
        """
        escaped_search_term = search_term.replace(' ', '+')

        google_url = 'https://www.google.com/search?q={}&num={}&hl={}' \
                     '&lr={}&cr={}&gl{}'\
            .format(escaped_search_term, number_results+1, language_code,
                    lr, cr, gl)
        response = get(google_url, headers=usr_agent)
        response.raise_for_status()

        return response.text

    def parse_results(raw_html):
        """
        Extract URLs from HTML response.
        """
        soup = BeautifulSoup(raw_html, 'html.parser')
        result_block = soup.find_all('div', attrs={'class': 'g'})
        for result in result_block:
            link = result.find('a', href=True)
            title = result.find('h3')
            if link and title:
                yield link['href']

    html = fetch_results(term, num_results, lang)
    return list(parse_results(html))


def keywords_per_site(site_list):
    """
    Calculate keywords appearing on given site.
    Save results in kwords_db folder per site.
    """
    _, output_dir = dir_from_project('iterative')
    for site in site_list:
        finder = Site_keyword(site, output_dir)
        all_kwords = finder.all_keyword()
        kwords_dir = '../kwords_db'
        file = '%s_kw.json' % site
        data = {}
        data['language'] = ''  # pass from lang detect
        data['name'] = '%s' % site
        data['len_all_kwords'] = len(all_kwords)
        data['all_kwords'] = sorted(str(el) for el in all_kwords)

        with open(path_join(kwords_dir, file), 'w') as outfile:
            json.dump(data, outfile, indent=4, sort_keys=False)
            outfile.close()


def keywords_count_per_iteration():
    """
    Count how frequently a keyword appears between the websites within one
    iteration. Return the result in descending order.
    """
    input_dir = '../kwords_db'
    all_key = []
    for f in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, f)) and 'json' in f:
            with open(path_join(input_dir, f), 'r') as json_file:
                data = json.load(json_file)
                for key in data:
                    if key == 'all_kwords':
                        site_kw = [el for el in data[key]]
                        all_key += site_kw
    count = Counter(word for word in all_key)
    keyword_list = []
    for value, count in count.most_common():
        keyword_list.append(value)
    list_to_json('../iterative', 'keyword_list.json', 'keyword_list',
                 keyword_list)
