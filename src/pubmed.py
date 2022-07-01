#!/usr/bin/env python3

__authors__ = ( 'Fabio Cumbo (fabio.cumbo@unitn.it)' )
__version__ = '0.01'
__date__ = 'Sep 09, 2021'

import sys, os, requests
from bs4 import BeautifulSoup
import argparse as ap

__DOMAIN__ = "pubmed.ncbi.nlm.nih.gov"
__DOMAINURL__ = "{}{}".format("https://", __DOMAIN__)
__BASEURL__ = "{}{}".format(__DOMAINURL__, "/?term=")

__HEADERS__ = {
    "Host": __DOMAIN__,
    "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8"
}

def read_params():
    p = ap.ArgumentParser( description = ( "The pubmed.py script searches for the number of papers that match with keywords." ),
                           formatter_class = ap.ArgumentDefaultsHelpFormatter )
    p.add_argument( '--keywords', 
                    type = str,
                    help = "Comma separated list of keywords." )
    p.add_argument( '--filelist', 
                    type = str,
                    help = ( "Path to the file with the list of keywords. Header must be specified in the first line. "
                             "It could be tab separated values if multiple columns are provided" ) )
    p.add_argument( '--indexby', 
                    type = str,
                    help = "Search for keywords under the specified column." )
    p.add_argument( '-v', 
                    '--version', 
                    action = 'version',
                    version = 'pubmed.py version {} ({})'.format( __version__, __date__ ),
                    help = "Print the current pubmed.py version and exit" )
    return p.parse_args()

def build_url(baseurl, keywords):
    return "{}{}".format(baseurl, requests.utils.requote_uri(" AND ".join(["({}[Title/Abstract])".format(key) for key in keywords])))

def get_results(baseurl, keywords):
    url = build_url(baseurl, keywords)
    request = requests.get(url, headers=__HEADERS__)
    soup = BeautifulSoup(request.content, "lxml")
    try:
        hits = soup.find_all("span", {"class": "value"})
        return int(hits[0].get_text().strip().replace(".", "").replace(",", ""))
    except:
        #raise Exception("\nUnable to retrieve the number of results for the following keywords:\n{}".format(keywords))
        return 0

if __name__ == '__main__':
    # Init params
    args = read_params()
    
    if args.keywords:
        keywords = args.keywords.lower().split(",")
        hits = get_results(__BASEURL__, keywords)
        print(hits)
    elif args.filelist and args.indexby:
        with open(args.filelist) as infile:
            header = list()
            for line in infile:
                line = line.strip()
                if line:
                    if line.startswith("#"):
                        header = line[1:].strip().split("\t")
                    else:
                        line_split = line.split("\t")
                        keywords = line_split[header.index(args.indexby)].lower().split(",")
                        hits = get_results(__BASEURL__, keywords)
                        print("{}\t{}".format(line, hits))
