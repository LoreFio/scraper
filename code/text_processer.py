# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:04:55 2020

@author: asys
@author: lfiorentini
@author: rcorsini
version: 1.1
"""

# module to perform NLP
import nltk
# module to perform NLP
import spacy
# class to tokenize strings with regexp
from nltk.tokenize import RegexpTokenizer
# class to tokenize strings with the nltk built-in word tokenizer
from nltk.tokenize import word_tokenize
# import Snowball stemmer to stem words
from nltk.stem.snowball import SnowballStemmer
# import lemmatizer
from nltk.stem import WordNetLemmatizer
# import list of english stopwords in order to strip them out later
from nltk.corpus import stopwords
# import pandas for a nicely formatted output
import pandas as pd

class Text_processer:
    def __init__(self, using_re_token = True, normalizer = "spacy",
                 language = 'english'):
        """
        The class constructor.

        Parameters :
        
        using_re_token : bool
            if true we'll use the RegexpTokenizer,
            otherwise the word_tokenize from nltk will be used instead.
        normalizer : str, optional
            string that defines the text normalizer to be used.
            The default is "spacy".
            Possible values: [WNLemmatizer, SnowballStemmer, spacy].
        language : str, optional
            language for the text normalizer. The default is "english".
            Possible values: ["english", "french"]
    
        Returns  :
            None
        """
        self.__using_re_token__ = using_re_token
        if self.__using_re_token__:
            self.__tokenizer__ = RegexpTokenizer(r'\w+').tokenize
        else:
            self.__tokenizer__ = word_tokenize
        
        self.__processer__ = None

        if normalizer == "WNLemmatizer" and language == 'english':
            self.__normalizer__ = WordNetLemmatizer().lemmatize
        elif normalizer == "SnowballStemmer":
            self.__normalizer__ = SnowballStemmer(language).stem
        elif normalizer == "spacy" and language == 'english':
            self.__normalizer__ = None
            self.__processer__ = spacy.load("en_core_web_md",
                                            disable=["parser", "ner"])
        elif normalizer == "spacy" and language == 'french':
            self.__normalizer__ = None
            self.__processer__ = spacy.load("fr_core_news_md",
                                            disable=["parser", "ner"])

        #elif normalizer == "WNLemmatizer" and language == 'french':
        #    self.__normalizer__ = SnowballStemmer(language).stem

        self.__stemmer__ = SnowballStemmer(language)
        self.__stopwords__ = stopwords.words(language)
        if language != 'english':
            self.__stopwords__ += stopwords.words('english')
        # if we are using spacy we add to the current list of stopwords the one
        # spacy ones
        if normalizer == "spacy":
            self.__stopwords__ = set(tok.lemma_ for tok in self.__processer__(
                ' '.join(self.__stopwords__)))
            self.__stopwords__.update(self.__processer__.Defaults.stop_words)

    def process(self, input_txt, any_num = True):
        """
        This function process the content using the tokenizer, and normalizer
        chosen in the constructor, if spacy is used the task are done at the
        same time. This code also remove stopwords and words with numbers

        Parameters :
        
        input_text : str
            List of texts to be analyzed.
        any_num : bool
            if True string with at least a number will be removed.
            if False string with only numbers inside will be removed.

        Returns :
        
        processed text.
        """
        
        if self.__normalizer__ != None: #using NLTK
            tokenized_list = self.__tokenize__(input_txt.lower())
            return self.__word_normalization__(self.remove_numbers(
                tokenized_list, any_num))
        return self.remove_numbers(
            [tok.lemma_ for tok in self.__processer__(input_txt.lower())
             if tok.lemma_ not in self.__stopwords__ and tok not in
             self.__stopwords__], any_num)
        

    def __tokenize__(self, input_txt):
        """
        This function tokenizes the content using the tokenizer chosen in the constructor.

        Parameters :
        
        input_text : str
            List of texts to be analyzed.

        Returns :
        
        tokenized input_txt.
        """
        return [elem for elem in self.__tokenizer__(input_txt) if elem not in
                self.__stopwords__]

    def remove_numbers(self, input_list, any_num = True):
        """
        This function removes strings containing any/only number(s) from a list
        of strings.
        
        Parameters :
        
        input_list : list
            list of texts to be analyzed.
        any_num : bool
            if True string with at least a number will be removed.
            if False string with only numbers inside will be removed.

        Returns :

        input_list without numbers.
        """
        if any_num:
            return [elem for elem in input_list if not any(char.isdigit()
                                                           for char in elem)]
        return [elem for elem in input_list if not elem.isdigit()]                            
    
    def __word_normalization__(self, input_list):
        """
        This function normalizes the tokenized content using the method chosen
        in the constructor.
        
        Parameters :
        
        input_list : list
            List of texts to be analyzed.
        
        Returns :
        
        input_lists after normalization.
        """
        return [self.__normalizer__(elem) for elem in input_list]
    
    def print_top_tab(self, wordlist, topnum = -1):
        """
        This function creates a Dataframe from a wordlist containg all the words
        in the list and their corresponding frequencies.
        
        Parameters :
        
        wordlist : list
            list of words.
        topnum : int
            if <=0 the dataframe will contain all the words,
            otherwise it will contain only the topnum most common words.

        Returns :
        
        DataFrame with frequency given in percentage.
        """
        fdist = nltk.FreqDist(wordlist)
        key = 'word'
        len_for_percentage = len(fdist)
        # put the two columns from the frequency distribution in a dataframe
        if topnum > 0:
            df = pd.DataFrame(fdist.most_common(topnum), columns =[key, 'count itt'])
            # add the frequency percentage in the document
            df['freq % itt'] = df['count itt']/len_for_percentage
            return df
        df = pd.DataFrame(list(fdist.items()), columns =[key, 'count itt'])
        # add the frequency percentage in the document
        df['freq % itt'] = df['count itt']/len_for_percentage
        return df