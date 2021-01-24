# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:59:00 2020

Spider adopted to precisely scrape 3 below HTML elements.
Crucial part is start_requests method which together with modified parse_page
call for 'div' element surpasses the JavaScript dynamically loaded content.

@author: asys, esavini
version 1.0.0
"""

import scrapy
from scrapy.spiders import CrawlSpider
from global_var_nlp import  path_join
import json


class MySpider(CrawlSpider):
    input_dir = '../input'
    encoding = 'utf8'
    passing_file = path_join(input_dir, 'site_to_scrap.txt')
    with open(passing_file, 'r', encoding = encoding) as f:
        input_file = str(f.readlines()[0]).replace('\n', '')
        f.close()

    with open(path_join(input_dir, input_file + '.json'), 'r') as json_file:
        data = json.load(json_file)
        json_file.close()
    name = data['name']
    allowed_domains = data['allowed_domains']
    start_urls = data['start_urls']

    def start_requests(self):
        '''
        This method fetches a response from the original url.
        '''
        for url in self.start_urls:
            yield scrapy.Request(url=url, callback=self.parse_page)

    def parse_page(self, response):
        for a in response.css('a'):
            yield {
                'span': a.css('a::text').get(),
            }
        for small in response.css('small'):
            yield {
                'p': small.css('small::text').get(),
            }
        for div in response.xpath('//div[@class = "media-body"]'):
            yield {
                'div': div.css('div::text').extract()[1],
            }

        print("URL: " + response.url)  # print visited urls to console

        # write visited urls into a file
        output_dir = '../output'
        file = 'visited_urls.txt'
        with open(path_join(output_dir, file), 'a') as outfile:
            print(response.url, file=outfile)
            outfile.close()


