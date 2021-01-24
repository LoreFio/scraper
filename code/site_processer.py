# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:25:11 2020

@author: lfiorentini
version: 1.1
"""

import json
import pickle
from os.path import join as path_join# import system for path definition
from os.path import exists as path_exists
import os
import langdetect
from gensim import corpora, models#, similarities
from langdetect import detect

from sentences_result import Sentences_result
from separate_lists import Separate_lists
from noise_filter import Noise_filter
from text_processer import Text_processer
import pandas as pd

class Site_processer:
    def __init__(self, name, output_dir, sub_dict, using_re_token = True,
                 normalizer = "spacy", language = 'english'):
        """
        The class constructor        

        Parameters :
        
        name : str
            Name of the site we are going to scrape.
        output_dir : str
            Name of the directory in which we will save outputs.
        sub_dict : dict
            Dictonary of regex used by the filter.
        using_re_token : bool, optional
            Kind of tokenizer used. The default is True.
        normalizer : str, optional
            string that defines the text normalizer to be used.
            The default is "spacy".
            Possible values: ["WNLemmatizer", "SnowballStemmer", "spacy"].
        language : str, optional
            language for the text processor. The default is "english".
            Possible values: ["english", "french", None].
            If "auto" then the language is obtained in function of the site
            content

        Returns :

        None.

        """
        self.__name__ = name
        self.__output_dir__ = path_join(output_dir, name)
        self.__language__ = language
        self.__using_re_token__ = using_re_token
        self.__normalizer__ = normalizer
        self.__filter__ = Noise_filter(sub_dict)
        #create output directory if not existing
        if not path_exists(self.__output_dir__):
            os.system('mkdir ' + self.__output_dir__)
        if self.__language__ != None:
            self.__processer__ = Text_processer(self.__using_re_token__,
                                                self.__normalizer__,
                                                self.__language__)
        self.__dictionary__ = None
        self.__corpus__ = None
    
    def open_n_divide(self, verbose = False):
        """
        This method open the json file, divide its content in the different
        parts and save them using the Separate_lists
        Parameters :
        
        verbose : bool, optional.
            Verbose option to pass to the divider. The default is False.

        Returns :
        
        None.

        """
        try:
            with open(path_join(self.__output_dir__, 'store.json')) as f:
                try:
                    json_dict = json.load(f)
                except json.decoder.JSONDecodeError: 
                    json_dict = {}
                f.close()
        except FileNotFoundError: 
            json_dict = {}
    
        Divider = Separate_lists(json_dict, self.__output_dir__)
        Divider.divide_save(verbose)
    
    def __load__(self, extension = "p_lists.txt"):
        """
        This class simply read a file and return the list of words

        Parameters :
        
        extension : str, optional
            DESCRIPTION. The default is "p_lists.txt".

        Returns :
        
        list of words in the file.

        """
        with open(path_join(self.__output_dir__, extension),
                  encoding = "utf8") as f:
            tmp_list = f.readlines()
            f.close()
        return tmp_list
    
    def __get_language__(self, p_list, header_list):
        """
        This method get the language applying langdetect to the first min_len
        lines of the p and header sections.


        Parameters
        ----------
        p_list : list
            list of text in the p section.
        header_list : list
            list of text in the header sections.

        Returns
        -------
        str
            language detected.

        """
        min_len = 10
        if max(len(p_list), len(header_list)) < min_len:
            print('site too small to detect language')
            return 'english'
        fr_counter = 0
        for i in range(min_len):
            try:
                fr_counter += detect(p_list[i]) == 'fr'
                fr_counter += detect(header_list[i]) == 'fr'
            except IndexError:
                pass
        if fr_counter >= (min_len / 4):
                return 'french'
#        print('fr_counter: ', fr_counter)
        return 'english'

    def __proc_text__(self, filtered_list, any_num = True):
        """
        This method unify the list, tokenize it, remove numbers, stopwords
        and finally normalize the words

        Parameters :
        
        filtered_list : list
            List filtered by the self.__filter__.
        any_num : bool
            any_num variable to be passed to the remove_numbers method of the
            processer
        Returns :
        
        normalized : list
            List normalized using processer.

        """
        return [self.__processer__.process(el, any_num)
                for el in filtered_list]
    
    def __save_sent__(self, sentences, filename):
        """
        This method convert a list of sentences into a list of tokens

        Parameters :
        
        sentence : Sentences_result
            sentences of normalized tokens.
        filename : str
            name of the file
            
        Returns :
        
            None.
        """
        with open(path_join(self.__output_dir__, filename),
                  'wb') as output:
            pickle.dump(sentences, output, pickle.HIGHEST_PROTOCOL)
        
    def __sent2list__(self, sentences):
        """
        This method convert a list of sentences into a list of tokens

        Parameters :
        
        sentence : Sentences_result
            sentences of normalized tokens.

        Returns :
        
        res : list
            list of normalized tokens.

        """
        res = []
        for lst in sentences.list_list:
            for el in lst:
                res.append(el)
        return res
        
    def __rename__(self, data, new_str):
        """
        This method rename the dataframe using new_str        

        Parameters :
        
        data : Dataframe
            dataframe to be renamed.
        new_str : str
            extension to be added to c_ and f_.

        Returns :
        
        None.

        """
        data.rename(columns = {'count itt': 'c_{}'.format(new_str),
                               'freq % itt': 'f_{}'.format(new_str)},
                    inplace = True)
        
    def in_top_bigrams(self, word, bigrams, key, percentage):
        """
        This method look for a word in the top elements of a dataframe column 

        Parameters :
        
        word : str
            word we look for in the list of top bigrams
        bigrams : Dataframe
            Dataframe in which we are going to look for.
        key : str
            Column of the dataframe we are going to use.
        percentage : double
            Between 0 and 1. Define the percentage of keywords considered
            relevant for each list.
        Returns :
        
        bool
            True if the word is in the top percentage bigrams, False otherwise.
        """
        if len(bigrams) == 0:
            return False
        
        for big in bigrams[bigrams[key] >= bigrams[key].quantile(
                1-percentage)]['bigram'].any():
            if word in big:
                return True
        return False
        
    def __create_dictionary__(self, norm_list):
        """
        This method create a dictionary (aka an hashmap) for all the words
        contained in the website

        Parameters :
                
        norm_list : list 
            list containing the different normalized contents.

        Returns :
            None
        """
        if self.__dictionary__ == None:
            self.__dictionary__ = corpora.Dictionary(norm_list)
            self.__dictionary__.save(path_join(self.__output_dir__,
                                               'gensim_dict.dict'))
        
    def __create_corpus__(self, norm_list):
        """
        This method create from all the sections contained in the website a
        bag-of-words corpus: format = list of (token_id, token_count)

        Parameters :
                
        norm_list : list 
            list containing the different normalized contents.

        Returns :
            None

        """
        self.__create_dictionary__(norm_list)
        if self.__corpus__ == None:
            self.__corpus__ = [self.__dictionary__.doc2bow(sec_list)
                               for sec_list in norm_list]
            corpora.MmCorpus.serialize(path_join(self.__output_dir__,
                                                'corpus.mm'),
                                       self.__corpus__)

    def __TF_IDF_score__(self, norm_list, smartirs = 'nnc'):
        """
        This method create a dictionary and a tfidf model for the 

        Parameters :
                
        norm_list : list 
            list containing the different normalized contents.
        smartirs : string
            string containing smartirs. More information available at
            https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System

        Returns :
        
        pd.DataFrame : pd.DataFrame
            DataFrame containing the word and the tfidf score according to the
            formula precised in the smartirs.

        """
        if len(norm_list) == 0:
            return pd.DataFrame(columns = ['word', smartirs])
        self.__create_corpus__(norm_list)
        tfidf = models.TfidfModel(self.__corpus__, smartirs = smartirs)
        return pd.DataFrame({self.__dictionary__.get(id): value
                             for doc in tfidf[self.__corpus__]
                             for id, value in doc}.items(),
                            columns = ['word', smartirs])
    
    def load_n_normalize(self, any_num = True):
        """
        This method open the .txt files created by open_n_divide, filter their
        contents and return a list with the different normalized contents
        
        Parameters :
            
        any_num : bool
            any_num variable to be passed to the remove_numbers method of the
        
        Returns :
        
        norm_list: list
            list containing the different normalized contents.
        """

        # Load and filter lists of words

        filtered_p = self.__filter__.filter_re(self.__load__("p_lists.txt"))
        filtered_span = self.__filter__.filter_re(self.__load__("span_lists.txt"))
        filtered_header = self.__filter__.filter_re(self.__load__(
            "header_lists.txt"))

        if self.__language__ == None:
            try:
                self.__language__ = self.__get_language__(filtered_p,
                                                          filtered_header)
            except langdetect.lang_detect_exception.LangDetectException:
                self.__language__ = "english"
            self.__processer__ = Text_processer(self.__using_re_token__,
                                                self.__normalizer__,
                                                self.__language__)

        sent_p = Sentences_result(self.__proc_text__(filtered_p, any_num))
        sent_span = Sentences_result(self.__proc_text__(filtered_span,
                                                        any_num))
        sent_header = Sentences_result(self.__proc_text__(filtered_header,
                                                          any_num))
        
        self.__save_sent__(sent_p, "sentences_p.pkl")
        self.__save_sent__(sent_span, "sentences_span.pkl")
        self.__save_sent__(sent_header, "sentences_header.pkl")
        
        norm_p = self.__sent2list__(sent_p)
        norm_span = self.__sent2list__(sent_span)
        norm_header = self.__sent2list__(sent_header)
        
        norm_list =(norm_p, norm_span, norm_header)
        return norm_list 
    
    def create_table(self, norm_list, smartirs_list, percentage):
        """
        This method create the data that resume the website content saving
        them in a proper file starting from the normalized text and a
        percentage
        
        Parameters :
        
        norm_list : list 
                    list containing the different normalized contents.
        smartirs_list : list 
                    list containing the different smartirs to compute tfidf
                    scores.
        percentage : double
            Between 0 and 1. Define the percentage of keywords considered
            relevant for each list.
        Returns :
        
        content_data: Dataframe
            A pandas Dataframe resuming the main information about the site.
        """

        word_tab_p = self.__processer__.print_top_tab(norm_list[0])
        word_tab_span = self.__processer__.print_top_tab(norm_list[1])
        word_tab_header = self.__processer__.print_top_tab(norm_list[2])
        
        """
        Commented out since it has not proven to be useful. It has not been
        deleted since it may provide some interesting results in the future
        # save n_grams freq dist
        for n in range(2,6):
            n_df = self.__processer__.n_grams(norm_list[0], n)
            n_df.to_csv(path_join(self.__output_dir__,
                                  '{}_gram.csv'.format(n)), index = False)
        """
        # we rename the columns in order to avoid conflicts in the next merge
        self.__rename__(word_tab_p,'p')
        self.__rename__(word_tab_span,'span')
        self.__rename__(word_tab_header,'header')

        # we merge the dataframes to have all the 
        content_data = word_tab_p.merge(
            right = word_tab_span, how = 'outer', on = 'word').merge(
                right = word_tab_header, how = 'outer', on = 'word')

        # we replace the nan with 0 since they come from missing words in list
        content_data.fillna(value = 0, inplace=True)

        """
        we add the tfidf scores of the smartirs in smartirs_list
        we must pay attention to the fact that some tfidf scores cannot be
        computed or does not make sense for empty lists.
        """
        
        norm_list_reduced = [el for el in norm_list if len(el)>0]
        for smartirs in smartirs_list:
            content_data = content_data.merge(
                right = self.__TF_IDF_score__(norm_list_reduced, smartirs),
                how = 'outer', on = 'word')
        #this column will contain the total number of occurrencies in the site
        content_data['c_BOW'] = content_data['c_p'] + content_data['c_header']
        return content_data
    
    def save_table(self, data, extra_name = ''):
        """
        This function save a dataframe in the correct directory
    
            Parameters :
            
            data : pd.Dataframe
                dataframe to be save.
            extra_name : str, optional
                extra string to be added to the name. The default is ''.
    
            Returns :
            
            None.
    
        """
        if extra_name == '':
            data.to_csv(path_join(self.__output_dir__, 'frequency.csv'),
                        index = False)
        else:
            data.to_csv(path_join(self.__output_dir__,
                          'frequency_' + extra_name + '.csv'), index = False)
            
    def corr(self, data):
        """
        This function compute, save and return the correlation of the dataframe
    
            Parameters :
            
            data : pd.Dataframe
                dataframe to be save.
    
            Returns :
            
            Q1 : pd.Dataframe.
                correlation matrix
    
        """
        
        Q = data.drop([col for col in data.columns 
                       if 'c_' in col or 'top_' in col], axis = 1).corr()
        Q.to_csv(path_join(self.__output_dir__, 'corr.csv'),
                 index = False)
        return Q