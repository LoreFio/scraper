#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:41:57 2020

@author: lfiorentini
"""

from math import isnan
from os.path import join as path_join
import pandas as pd
from scipy import spatial

class Site_similarity():
    def __init__(self, output_dir, site_1, site_2):
        """
        Constructor for the similarity comparisons

        Parameters
        ----------
        output_dir : str
            path to the output directory.
        site_1 : str
            first site name.
        site_2 : str
            second site name.

        Returns
        -------
        None.

        """
        try:
            with open(path_join(output_dir, site_1, 'frequency.csv'),
                      encoding="utf8") as f:
                data_1 = pd.read_csv(f)
                f.close()
        except FileNotFoundError:
                data_1 = pd.DataFrame(data = None,
                                      columns = ['word', 'c_p', 'f_p',
                                                 'c_span', 'f_span',
                                                 'c_header', 'f_header', 'nnc',
                                                 'nnn', 'anc', 'c_BOW'])
        try:
            with open(path_join(output_dir, site_2, 'frequency.csv'),
                      encoding="utf8") as f:
                data_2 = pd.read_csv(f)
                f.close()
        except FileNotFoundError:
                data_2 = pd.DataFrame(data = None,
                                      columns = ['word', 'c_p', 'f_p',
                                                 'c_span', 'f_span',
                                                 'c_header', 'f_header', 'nnc',
                                                 'nnn', 'anc', 'c_BOW'])
        reduced_cols = ['word', 'c_p', 'c_span', 'c_header', 'c_BOW']
        data_1 = data_1[reduced_cols]
        data_2 = data_2[reduced_cols]
        data_1.columns = ['word'] + [col + '_1' for col in
                                              data_1.columns if
                                              col != 'word']
        data_2.columns = ['word'] + [col + '_2' for col in
                                              data_2.columns if
                                              col != 'word']
        self.__merged__ = data_1.merge(right = data_2,
                                                how = 'outer', on = 'word')
        self.__merged__.fillna(value = 0, inplace=True)
        
    def __cosine_similarity__(self, col):
        """
        Computes the cosine similarity of the two BOW vectors using the column
        col if 'word' is passed as a col it returns 0 

        Parameters
        ----------
        col : str
            column name in the original frequency file.

        Returns
        -------
        float
            cosine similarity.

        """
        if col != 'word':
            return 1-spatial.distance.cosine(self.__merged__[col + '_1'],
                                             self.__merged__[col + '_2'])
        return 0
    
    def total_similatiry(self):
        """
        Computes the cosine similarity of the two BOW vectors using the total
        number of occurrencies of the words in the site

        Returns
        -------
        float
            cosine similarity.

        """
        return self.__cosine_similarity__('c_BOW')
    
    def avg_similatiry(self):
        """
        Computes the average cosine similarity of the two BOW vectors of each 
        section
        
        Returns
        -------
        float
            cosine similarity.

        """
        res = 0
        counter = 0
        tmp = self.__cosine_similarity__('c_p')
        #we need to handle the case in which there is not a section
        if not isnan(tmp):
            counter += 1
            res += tmp
        tmp = self.__cosine_similarity__('c_span')
        if not isnan(tmp):
            counter += 1
            res += tmp
        tmp = self.__cosine_similarity__('c_header')
        if not isnan(tmp):
            counter += 1
            res += tmp
            
        return res/counter