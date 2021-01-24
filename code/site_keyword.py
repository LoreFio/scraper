#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:10:28 2020

@author: lfiorentini
"""

from os.path import join as path_join
import pandas as pd

class Site_keyword():
    def __init__(self, site, output_dir, n_keyword = 30,
             useless_w_list = []):
        """
        The class constructor        
    
        Parameters :
        
        site : str
            site to consider.
        output_dir : str
            Name of the directory in which we will save outputs.
        n_keyword : int, optional
            number of keyword that we are going to consider for each model.
            The default is 50
        useless_w_list : list, optional
            Kind of tokenizer used. The default is True.
        Returns :
    
        None.
    
        """
        
        self.__site__ = site
        self.__output_dir__ = output_dir
        self.__n_keyword__ = n_keyword
        self.__useless_w_list__ = set(useless_w_list)
        self.__data___dict__ = {}
        self.__data____ = None
        self.__sure_key_site__ = {}
        
        with open(path_join(self.__output_dir__, site, 'frequency.csv'),
                  encoding="utf8") as f:
            self.__data__ = pd.read_csv(f)
            f.close()
        self.__col_list__ = [col for col in self.__data__.columns
                             if 'c_' not in col]
        self.__col_list__.remove('f_span')
        '''
        We consider both frequencies and smartirs for tfidf.
        We don't consider the span frequency since it's often inefficient.
        '''
    def extract_top(self, site, col):
        """
        This method find the top elements for a given site and column

        Parameters :
    
        col : string
            column of which we want the top elements.

        Returns :
        
        list
            list of top elements.

        """
        return list(self.__data__.sort_values(
            col, ascending=False)['word'].head(self.__n_keyword__))
    
    def all_keyword(self):
        """
        This method returns all the keywords from the list of website grouped

        Parameters :
            None.

        Returns :
            list
                list of all keywords.

        """
        
        resu = []
        for col in self.__col_list__:
            resu += self.extract_top(self, col)
        return resu