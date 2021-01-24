# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:27:13 2020

Module to apply LinkExtractor rules in broad crawl. It tells the spider which
HTML nodes to follow (allow) and which he should not enter (deny). Some of them
are loaded from iterative search, some are specified here.

@author: lfiorentini, esavini, asys
version 1.4
"""

from os.path import join as path_join
import json
from global_var_nlp import dir_from_project


class Save2Json:

    def __init__(self, google_name, google_url, top_level_domain):
        """
        Save the rules into an input json file.

        Parameters :

        common_deny : list
            Common nodes not to follow.
        languages_deny: list
            Language codes to omit during scrape.
        specific_deny : list
            Insert particular HTML nodes below to exclude them from broad crawl
            Precise use.
        name : str
            Name of the file.
        allowed_domains : url
            Website which we want to scrape. ("example.com")
        start_urls : url
            Initial links for broad crawl.
        restrict_text : list
            Allow only the content which contains given text. Applied within
            items (eg. a paragraph). For precise use.

        Returns :

        input file : json
            Contains all the rules obtained from iterative search and specified
            in this module.
        """
        self.google_name = google_name
        self.google_url = google_url
        self.top_level_domain = top_level_domain

        input_dir,_ = dir_from_project('iterative')
        file = '%s.json' % self.google_name

        common_deny = ['contact', 'newsletter', 'special-offers',
                       'financial-services', 'find-a-dealer', 'site-map',
                       'forum', 'privacy', 'terms', 'faq', 'finance',
                       'careers', 'contact-us', 'gallery', 'healthcare',
                       'health-care', '/events/']
        languages_deny = ['/es/', '/de/', '/at/', '/be/', '/cz/',
                          '/ie/', '/it/', '/no/', '/pl/', '/se/', '/sk/',
                          '/ja/', '/ko/', '/bs/', '/pt/', '/bg/', '/zh/',
                          '/cs/', '/da/', '/ro/', '/ru/', '/sq/', '/sr/',
                          '/nl/', '/nb/', '/sk/', '/sl/', '/sv/', '/tr/',
                          '/hu/', '/hr/', '/mk/', '/ua/']  #'/fr/', '/en/'
        specific_deny = []

        data = {}
        data['language'] = ''  # pass from langdetect
        data['name'] = '%s' % self.google_name
        data['allowed_domains'] = ['%s' % self.top_level_domain]
        data['start_urls'] = ['%s' % self.google_url]
        '''
        common_allow = ['auto-industry', 'auto-manufacturer', 'auto-suppliers',
                         'automobile', 'automotive', 'autonomous', 'car',
                         'car-makers', 'carbon-offset', 'carmakers',
                         'charging infrastructure', 'decarbonization',
                         'emerging-trends-real-estate', 'energy-demand',
                         'energy-insights', 'ev adoption',
                         'innovation', 'low carbon', 'market-intelligence',
                         'mobility', 'mobility ecosystem', 'mobilitys',
                         'oil-and-gas', 'resource-revolution',
                         'sectors-without-borders', 'self-driving',
                         'sustainability', 'sustainable', 'transport',
                         'vehicle']
        specific_allow = []
        data['allow'] = common_allow + specific_allow
        '''
        data['allow'] = []
        data['deny'] = languages_deny + specific_deny  # + common_deny
        # data['deny_domains'] = ()
        # data['restrict_text'] = []

        with open(path_join(input_dir, file), 'w') as outfile:
            json.dump(data, outfile, indent=4, sort_keys=False)
            outfile.close()

        """
        For reading
        with open(path_join(input_dir, file), 'r') as json_file:
            data = json.load(json_file)
            print(data['name'])
            json_file.close()
        """