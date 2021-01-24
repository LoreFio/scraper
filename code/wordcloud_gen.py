# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:57:51 2020

@author: lfiorentini
"""

from os.path import join as path_join# import system for path definition
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from global_var_nlp import get_site_list

site_list = get_site_list()

for site in site_list:
    freq = pd.read_csv(path_join('../output', site, 'frequency.csv'))
    nnc_dit = dict(zip(freq['word'].tolist(), freq['nnc'].tolist()))
    wc = WordCloud(background_color="black", colormap="Greens",
                   max_words=70).generate_from_frequencies(nnc_dit)
    plt.imshow(wc, interpolation="bilinear")
    plt.title(site)
    plt.axis("off")
    plt.show()
    plt.savefig(path_join('../output', site, 'wordcloud.png'), bbox_inches='tight')
    plt.close()
    