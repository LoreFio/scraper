# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:03:48 2020

@author: lfiorentini
version 1.1
"""

def find_opportunities(data, bad_pos = 9, q = 0.90):
    """
    This function looks for keywords with a lot of Search Volume and a current 
    bad position

    Parameters
    
    data : pd.DataFrame
        input table.
    bad_pos : int64, optional
        position after which the keyword is useless. The default is 9.
    q : float64, optional
        Quantile of the Search Volume distribution. The default is 0.75.

    Returns
    
    pd.DataFrame
        Result table.

    """
    high_vol = data['Search Volume'].quantile(q)
    return data[(data['Search Volume'] > high_vol) &
                (data['Position'] > bad_pos)]
    
def contain_abbreviations(kword, site_abbreviations):
    """
    This function returns whether the keyword contains any site abbreviation or
    not

    Parameters
    
    kword : str
        keyword.
    site_abbreviations : list
        list of site abbreviations.

    Returns
    
    bool
        whether the keyword contains any site abbreviation.
        
    """
    for el in site_abbreviations:
        if el in kword:
            return True
    return False
        