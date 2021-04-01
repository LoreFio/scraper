# Scrapper
Python Scrapper for websites and NLP analysis

pip install -r requirements.txt
python -m spacy download en_core_web_md
python -m spacy download en_core_web_sm
python -m nltk.downloader all
Then it is necessary to dowload from git and prepare the directories
git clone https://codehub.tki.la/lfiorentini/scraper.git
cd scraper

Finally, if we want for example to analyse the site renault_onair (for which the json file is already present in the directory ./input) we have to execute the commands

mkdir output
mkdir kwords_db
mkdir iterative
mkdir logs
touch kwords_db/kw.json
cd code/
python3 launch_scrapy.py -f ces2020
python3 launch_scrapy.py -f ces2020
# it is not an error that script has to be launched twice
python3 scraper_pipeline.py -f ces2020
python3 wordcloud_gen.py -f ces2020

# Basic parameters and settings
In spider_results.py
change CLOSESPIDER_ITEMCOUNT to 250000 for production size scrape or keep it
between 1000 to 5000 for testing.
LOG_LEVEL is set to CRITICAL. Adjust it to ERROR, INFO or DEBUG when needed.

# Launching iterative process
In order to obtain a significant number of URLs where given keywords
can be found use below command and put the concerned keywords in
iterative/keyword_list.py file. Additionally in iterative_search.py you can
adjust the size of iterations by choosing how many keywords pass to next
iteration by changing index size in 'keys' variable (line 58). You can also
define keywords which will be used in every search combination by putting them
in 'must_have_keys' variable.

python3 iterative_process.py -i 10 -l 'en'
