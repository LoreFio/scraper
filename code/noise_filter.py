# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:45:38 2020

Script to filter noise from the scraped content.

@author: asys
@author: lfiorentini
version 1.1
"""

import re

class Noise_filter:
    def __init__(self, list_of_regex):
        """
        The class constructor.

        Parameters :
        
        list_of_regex : dict
            dictionary of regular expressions that will be applied.
            The key is the element we want to remove and the value - the replacer.
        Returns :
        
            None.
        """
        self.__list_of_regex__ = list_of_regex

    def filter_re(self, list_of_content):
        """
        This method filters the content using regular expressions.

        Parameters :
        
        list_of_content : list
            list containing results from the scraper.

        Returns :
        
        list containing all content elements after the list_of_regex was
        applied on list_of_content; with empty elements removed.
        """
        for key in self.__list_of_regex__:
            list_of_content = [re.sub(key, self.__list_of_regex__[key], file)
                             for file in list_of_content]
        # remove " " and empty string as well as duplicates
        to_remove = set()
        for elem in list_of_content:
            if len(elem) < 1 or elem == " ":
                to_remove.add(elem)
        return list(set(list_of_content)-to_remove)
