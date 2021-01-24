# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:52:21 2020

@author: lfiorentini, asys
"""
from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrapy.signalmanager import dispatcher

def spider_results(site, project = 'renault', out_file = 'out.json'):
    """
    Wrapper for launching Scrapy.
        
    Parameters :
    
    site : str
        Name of the site we are scraping from.
    project : str
        Name of the project we are working on. The default is 'renault'
    out_file : str
        Name of the file where we want to save the result. The default is
        out.json
    Returns :
    
        List of items (dictionaries) processed by the scraper

    """
    
    if project == 'renault':
        from broad_crawl_spider import MySpider
    elif project == 'iterative':
        from iterative_spider import MySpider
    else:
        print('No spider for project:', project)
        return
    results = []

    def crawler_results(signal, sender, item, response, spider):
        results.append(item)

    dispatcher.connect(crawler_results, signal=signals.item_passed)
    # Scrapy default_settings are overridden by below rules
    settings = get_project_settings()
    settings['ROBOTSTXT_OBEY'] = True
    settings['LOG_LEVEL'] = 'CRITICAL'
    settings['FEED_FORMAT'] = 'json'
    settings['FEED_URI'] = 'file:../output/%s/store.json' % site
    settings['CLOSESPIDER_ITEMCOUNT'] = 2000
    # 250000 for production use
    # 1000 to 5000 for testing
    settings['HTTPERROR_ALLOWED_CODES'] = [301]
    '''
    If you get HTTP error 403 - change USER_AGENT

    To activate Selenium use below setting:
    DOWNLOADER_MIDDLEWARES = {
        'mobility.mobility.scraper.code.selenium_mid.SeleniumMiddleware': 500
    }
    '''
    process = CrawlerProcess(settings)
    process.crawl(MySpider)
    process.start()  # the script will block here until the crawling is finished
    return results
