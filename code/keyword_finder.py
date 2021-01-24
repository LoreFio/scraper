# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:10:15 2020

@author: lfiorentini
version: 1.1
"""

from os.path import join as path_join
import numpy as np
import pandas as pd


class Keyword_finder:
    def __init__(self, site_list, output_dir, n_keyword = 30,
                 useless_w_list = []):
        """
        The class constructor        

        Parameters :
        
        site_list : list
            list of the sites we are going to consider.
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
        
        self.__site_list__ = site_list
        self.__n_site__ = len(self.__site_list__)
        self.__output_dir__ = output_dir
        self.__n_keyword__ = n_keyword
        self.__useless_w_list__ = set(useless_w_list)
        self.__Data_dict__ = {}
        self.__Data__ = None
        self.__sure_key_site__ = {}
        
        for site in self.__site_list__:
            # read the files
            with open(path_join(self.__output_dir__, site, 'frequency.csv'),
                      encoding="utf8") as f:
                self.__Data_dict__[site] = pd.read_csv(f)
                f.close()
        self.__col_list__ = [col for col in self.__Data_dict__[
            self.__site_list__[0]].columns if 'c_' not in col]
        self.__col_list__.remove('f_span')
        '''
        with this dirty trick we consider both frequencies and smartirs for
        tfidf.
        We don't consider the span frequency since it's often
        rich of useless stuff
        '''
        self.__create_Data__()
        
    def __create_Data__(self):
        """
        This method create inside self.__Data__ a dataframe where all the
        keywords from the websites are stored and with counters for their
        appearences in the most important indicators.
        
        Returns:
            None.
            
        """
        all_kwords = set()
        for site in self.__site_list__:
            for smartirs in self.__col_list__:
                all_kwords.update(self.extract_top(site, smartirs))
        all_kwords = list(all_kwords.difference(self.__useless_w_list__))
        
        n_words = len(all_kwords)
        n_cols = len(self.__col_list__)
        occur_mat = np.zeros([n_words, n_cols])
        for site in self.__site_list__:
            for i in range(n_words):
                for j in range(n_cols):
                    occur_mat[i,j] += int(all_kwords[i] in
                                          self.extract_top(site,
                                                      self.__col_list__[j]))
                    
        self.__Data__ = pd.DataFrame(occur_mat, columns = self.__col_list__)
        self.__Data__['word'] = all_kwords
        
    def extract_top(self, site, col):
        """
        This method find the top elements for a given site and column

        Parameters :
        
        site : string
            site of which we want the top elements.
.
        col : string
            column of which we want the top elements.

        Returns :
        
        list
            list of top elements.

        """
        return list(self.__Data_dict__[site].sort_values(
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

        return list(self.__Data__['word'])
    
    def common_keyword_col(self, col, percentage = 0.25):
        """
        This method returns all the keywords that appear at least with a 
        percentage among the files using col as criterium
        

        Parameters
        ----------
        col : str
            column that we are interested in.
        percentage : float, optional
            how many files must contain this word. The default is 0.25.

        Returns
        -------
        list
            list of filtered words.

        """
        return list(self.__Data__[self.__Data__[col] >=
                              percentage*self.__n_site__]['word'])

    def common_keyword(self, col_min, percentage = 0.25):
        """
        This method returns all the keywords that appear at least col_min times
        using common_keyword_col criterium on all the columns
        

        Parameters
        ----------
        col : int
            minimum number of column where the word has to be present.
        percentage : float, optional
            how many files must contain this word. The default is 0.25.

        Returns
        -------
        list
            list of filtered words.

        """
        count_dict = {}
        support_dict = {}
        for col in self.__Data__.columns:
            if col != 'word':
                support_dict[col] = self.common_keyword_col(col, percentage)
                for word in support_dict[col]:
                    if word in count_dict:
                        count_dict[word] += 1
                    else:
                        count_dict[word] = 1
        return [key for key in count_dict if count_dict[key] >= col_min],\
            support_dict
 
    def get_data(self):
        """
        This method simply saves a copy of the dataframe created

        Returns :
        
        pd.DataFrame
            a copy of the dataframe created.

        """
        return self.__Data__.copy()