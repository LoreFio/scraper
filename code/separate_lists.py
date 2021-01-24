# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:22:07 2020

Script to divide the extracted text content into separate lists using CSS section as a separator.

@author: asys
@author: lfiorentini
version: 1.1
"""

# import regular expressions (re) module
import re
# import system for path definition
from os.path import join as path_join

class Separate_lists:
    def __init__(self, content_as_a_list, directory):
        """
        The class constructor to divide content into separate lists.

        Parameters :
        
        content_as_a_list : str
            the scraper extracts all the content as one list.
        directory : str
            directory where we will save the results.

        Returns :
        
            None

        Note : in content_as_a_list we replace single quote to double quote
        (json formatting reasons).
        """
        self.content = str(content_as_a_list).replace("'", '"')
        self.__basic_re__ = r'\{\"key\"\: \"(.+?)\"\}'
        self.__section_list__ = ["header", "p", "span"]
        self.__directory__ = directory

    def __replacer__(self, new_key):
        """
        This method replaces the general regex with the correct CSS section.

        Parameters :
        
        new_key : str
            section we want to separate.
        Returns :
        
        the list of content in that section.
        """
        return re.findall(re.sub("key", new_key, self.__basic_re__),
                          self.content)

    def __writer__(self, curr_list, filename, encoding='UTF-8'):
        """
        This method writes the separate sections' lists in output files.

        Parameters :
        
        curr_list : list
            list of strings to be saved.
        filename : str
            filename that we are going to use.

        Returns :
        
        output files : each CSS section is saved in a separate txt file in
        outputs folder.
        """
        with open(filename, 'w', encoding=encoding) as filehandle:
            for listitem in curr_list:
                filehandle.write('%s\n' % listitem)
            filehandle.close()

    def divide_save(self, verbose=False):
        """
        This method divides content into separate CSS sections and saves it.

        Parameters :
        
        verbose : bool
            The default is False.

        Returns :
        
        Note : For each CSS section we cut out only the text content.

        """
        for section in self.__section_list__:
            self.__writer__(self.__replacer__(section),
                            path_join(self.__directory__,
                                      section) + "_lists.txt", 'UTF-8')

        self.__writer__(re.findall(r'(www[^\s]+)\"\}', self.content),
                        path_join(self.__directory__, "link_lists.txt"),
                        'UTF-8')

        if (verbose):
            for section in self.__section_list__:
                print(section)
                print(self.__replacer__(section))
                
            print("Links:")
            print(re.findall(r'(www[^\s]+)\"\}', self.content))