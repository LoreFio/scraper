# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:46:12 2020

Quantile values printing discarded for analysis readability.

@author: lfiorentini
version: 1.1
"""
import pandas as pd
from os.path import join as path_join # import system for path definition
# module to perform NLP
import spacy
from nltk.corpus import wordnet as wn  
from nltk.stem.snowball import SnowballStemmer
# import lemmatizer
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import pickle
import multiprocessing
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

class Semantic_search():
    def __init__(self, output_dir, site, percentage = 0.01,
                 language = "english", normalizer = "spacy"):
        """
        The class constructor        

        Parameters :
        
        output_dir : str
            Name of the directory in which we will save outputs.
        site: str
            Name of the site we are studying.
        percentage : float
            percentage to be considered as top. The default is 0.01.
        language : str, optional
            language of the content. The default is "english".
        normalizer : str, optional
            string that defines the text normalizer to be used.
            The default is "spacy".
            Possible values: [WNLemmatizer, SnowballStemmer, spacy].

        Returns :
        
        None.

        """        
        self.__output_dir__ = output_dir
        self.__site__ = site
        self.n_list = range(2, 6)
        self.__percentage__ = percentage
        self.__site_freq__ = pd.read_csv(path_join(self.__output_dir__,
                                                   self.__site__,
                                                   'frequency.csv'))
        self.__language__ = language
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
        
        with open(path_join(self.__output_dir__, self.__site__,
                            'sentences_p.pkl'), 'rb') as input_file:
             sentences = pickle.load(input_file)
             input_file.close()
        
        try:
            self.__model_p__ = Word2Vec(sentences.list_list, min_count = 2,
                                        size = 50, workers = 4)
        except RuntimeError:
            #occurs if paragraph is empty the we do it with div
            with open(path_join(self.__output_dir__, self.__site__,
                            'sentences_div.pkl'), 'rb') as input_file:
                sentences = pickle.load(input_file)
                input_file.close()
            self.__model_p__ = Word2Vec(sentences.list_list, min_count = 2,
                                        size = 50, workers =
                                        multiprocessing.cpu_count()-1)

    def __proc_word__(self, word):
        """
        This method process a word using the processer

        Parameters :
        
        word : str
            word to be studied.

        Returns :

        Processed word.
        """
        return [el.lemma_ for el in self.__processer__(word)][0]
    
    def is_keyword(self, word):
        """
        This method shows is a word is a keyword for the site

        Parameters :
        
        word : str
            word to be studied.

        Returns :

        None.
        """
        found = False
        if self.__normalizer__ != None:
            w = self.__normalizer__(word)
        else:
            w = self.__proc_word__(word)
        line = self.__site_freq__[self.__site_freq__['word'] == w]
        if len(line) == 0:
            print("The world", w, "has not been used in site", self.__site__)
            return

        for col in self.__site_freq__.columns:
            if 'c_' not in col and col != 'word':
                q = self.__site_freq__[col].quantile(1 - self.__percentage__)
                if float(line[col]) > q:
                    found = True
                    # print("word:", w, "col:", col, "quantile:", q,
                    #      "value:", float(line[col]))
        if found == False:
            print("The world", w, "appears but it is not key in site",
                  self.__site__)
        return
    
    def sem_sphere(self, word):
        """
        This method creates and saves the semantic sphere of a given word

        Parameters :
        
        word : str
            word to be studied.

        Returns :


        """
        if self.__language__ == "english":
            syn_set = set()
            for l in [s.lemma_names() for s in wn.synsets(word)]:
                syn_set.update(l)
            if self.__normalizer__ != None:
                used_syn = [self.__normalizer__(s) for s in syn_set if
                            self.__normalizer__(s) in
                            list(self.__site_freq__['word'])]
            else:
                used_syn = [self.__proc_word__(s) for s in syn_set if
                            self.__proc_word__(s) in
                            list(self.__site_freq__['word'])]
        elif self.__language__ == 'french':
            print('no synonims for french yet')
            used_syn = []
        
        try:
            if self.__normalizer__ != None:
                most_similar = self.__model_p__.wv.most_similar(
                    self.__normalizer__(word))
            else:
                most_similar = self.__model_p__.wv.most_similar(
                    self.__proc_word__(word))
        except KeyError: 
            most_similar = {}
        return used_syn, most_similar
    
    def clustering(self, k):
        word_vectors = self.__model_p__.wv
        KM_model = KMeans(n_clusters = k, max_iter = 1000, random_state = True,
                          n_init = 50).fit(X = word_vectors.vectors)

        center_closest = []
        for i in range(k):
            center_closest.append([el[0] for el in
                                   word_vectors.similar_by_vector(
                                       KM_model.cluster_centers_[i], topn = 15,
                                       restrict_vocab = None)])

        metric_str = 'euclidean'
        score = silhouette_score(word_vectors.vectors,
                                 KM_model.predict(word_vectors.vectors),
                                 metric = metric_str)
        print("silhouette_score:", score)

        SVmodel = SilhouetteVisualizer(KM_model, is_fitted = True)
        SVmodel.fit(word_vectors.vectors)
        SVmodel.show()  
        words = pd.DataFrame(word_vectors.vocab.keys(), columns = ['words'])
        words['vectors'] = words.words.apply(lambda x: word_vectors[f'{x}'])
        words['cluster'] = words.vectors.apply(lambda x: KM_model.predict(
            [np.array(x)]))
        words.cluster = words.cluster.apply(lambda x: x[0])
        words['closeness_score'] = words.apply(
            lambda x: 1/(KM_model.transform([x.vectors]).min()), axis = 1)

        return KM_model, center_closest, score, words