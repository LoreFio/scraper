#!/bin/bash

find ./output/ -name *.json -type f -delete

cd ./code/

folder="../input/"
ext=".json"
for f in ${folder}*${ext} 
do
	py_in="${f//..\/input\/}"
	py_in="${py_in//$ext/}"
	if [ $py_in != "site_to_scrap" ]
	then
		python launch_scrapy.py -f $py_in
		python launch_scrapy.py -f $py_in
	else
		echo "not processing $py_in"
	fi
done

cd ..
