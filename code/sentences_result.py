# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:15:57 2020

@author: lfiorentini
version: 1.1
"""

"""
Simple class used to save and load data with pickle
"""

class Sentences_result():
    def __init__(self, list_list = [[]]):
        """

        Parameters :
        
        list_list : list(list), optional
            list of other keywords if needed. The default is [[]].

        Returns :
        None.

        """
        self.list_list = list_list