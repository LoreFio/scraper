#!/bin/bash

find ./output/ -name corpus.* -type f -delete
find ./output/ -name *.dict -type f -delete
find ./output/ -name *.csv -type f -delete
find ./output/ -name *.txt -type f -delete
find ./output/ -name *.pkl -type f -delete

cd ./code/

python scraper_pipeline.py

cd ..
