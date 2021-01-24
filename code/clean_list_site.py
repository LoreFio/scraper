#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:01:56 2020

This functions remove sites of languages different from the chosen one
@author: lfiorentini
"""
from langdetect import detect
import pandas as pd
from os.path import join as path_join

def clean_list_site(output_dir, site_list, language):
    """
    This functions remove sites of languages different from the chosen one


    Parameters
    ----------
    output_dir : str
        output directory where we find the sites.
    site_list : list
        initial list of sites.
    language : str
        language we want to keep.

    Returns
    -------
    result : list
        filtered list.

    """
    result = []
    n_chunks = 10
    for site in site_list:
        counter = 0
        # read the files
        with open(path_join(output_dir, site, 'frequency.csv'),
                  encoding="utf8") as f:
            data = pd.read_csv(f)
            f.close()
            gen = chunks(list(data), n_chunks)
            for el in gen:
                my_str = ' '.join(el)                
                counter += (language == detect(my_str))
            if counter > 0.7 * n_chunks:
                result.append(site)
    return result

def chunks(l : list, n : int):
    """
    Yield n number of striped chunks from l.


    Parameters
    ----------
    l : list
        list to be striped.
    n : int
        integer.

    Yields
    ------
    generator
        Generator of chunks.

    """
    for i in range(0, n):
        yield l[i::n]
        