# -*- coding: utf-8 -*-
"""
Created on Thu Apr 09 12:06:00 2020

This spider is on hold. Further development needed.

@author: asys
version 1.0.0
"""
from mobility.mobility.scraper.code.selenium_mid import SeleniumRequest
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from os.path import join as path_join
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

    rules = (

        Rule(LinkExtractor(allow=data['allow'], deny=data['deny'],
                           deny_domains=data['deny_domains'],
                           restrict_text=data['restrict_text']),
             callback='parse_page', follow=True),

    )

    def start_requests(self):
        yield SeleniumRequest(self.start_urls[0], callback=self.parse)

    def parse_page(self, response):
        # following sections are combined in 'p' output
        # b, ul, li, strong
        for p in response.css('p'):
            yield {
                'p': p.css('p::text').get(),
            }
        for b in response.css('b'):
            yield {
                'p': b.css('b::text').get(),
            }
        for ul in response.css('ul'):
            yield {
                'p': ul.css('ul::text').get(),
            }
        for li in response.css('li'):
            yield {
                'p': li.css('li::text').get(),
            }
        for strong in response.css('strong'):
            yield {
                'p': strong.css('strong::text').get(),
            }
        for span in response.css('span'):
            yield {
                'span': span.css('span::text').get(),
            }
        for title in response.css('div.title'):
            yield {
                'header': title.css('div.title::text').get(),
            }
        # following sections are combined in 'header' output
        # head, h1, h2, h3, h4, h5, h6
        for head in response.css('head'):
            yield {
                'header': head.css('head::text').get(),
            }
        for h1 in response.css('h1'):
            yield {
                'header': h1.css('h1::text').get(),
            }
        for h2 in response.css('h2'):
            yield {
                'header': h2.css('h2::text').get(),
            }
        for h3 in response.css('h3'):
            yield {
                'header': h3.css('h3::text').get(),
            }
        for h4 in response.css('h4'):
            yield {
                'header': h4.css('h4::text').get(),
            }
        for h5 in response.css('h5'):
            yield {
                'header': h5.css('h5::text').get(),
            }
        for h6 in response.css('h6'):
            yield {
                'header': h6.css('h6::text').get(),
            }

        print("URL: " + response.url)  # print visited urls to console

        # write visited urls into a file
        output_dir = '../output'
        file = 'visited_urls.txt'
        with open(path_join(output_dir, file), 'a') as outfile:
            print(response.url, file=outfile)
            outfile.close()
