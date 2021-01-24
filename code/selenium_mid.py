# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:20:00 2020

@author: asys
version 1.1.0
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from scrapy import signals
from scrapy.http import HtmlResponse
from scrapy import Request


class SeleniumRequest(Request):
    pass


class SeleniumMiddleware:
    def __init__(self):
        self.driver = webdriver.Chrome(ChromeDriverManager().install())

    @classmethod
    def from_crawler(cls, crawler):
        middleware = cls()
        crawler.signals.connect(middleware.spider_closed, signals.spider_closed)
        return middleware

    def process_request(self, request, spider):
        if not isinstance(request, SeleniumRequest):
            return None
        self.driver.get(request.url)

        # loop to click "load more" button on JS-page
        while True:
            try:
                button = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'btn btn-secondary'))
                )
                self.driver.execute_script("arguments[0].click();", button)
            except:
                break

        return HtmlResponse(
            self.driver.current_url,
            body=str.encode(self.driver.page_source),
            encoding='utf-8',
            request=request
        )

    def spider_closed(self):
        self.driver.quit()
