#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import requests
import cv2
import os

from requests import exceptions

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# --query :The image search query we're using, which could be anything such as "pikachu", "santa" or "jurassic park"
# --output :The output directory for images
ap.add_argument("-q",
                "--query",
                required=True,
                help="search query to search Bing Image API for")
ap.add_argument("-o",
                "--output",
                required=True,
                help="path to output directory of images")
args = vars(ap.parse_args())

API_KEY = "7e350fa95d2b4b5994f9e6a6489d5906"
MAX_RESULTS = 1000
GROUP_SIZE = 50
# set the endpoint API URL
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"


EXCEPTIONS = set([
    IOError, FileNotFoundError, exceptions.RequestException,
    exceptions.HTTPError, exceptions.ConnectionError, exceptions.Timeout
])


term = args["query"]
headers = {"Ocp-Apim-Subscription-Key": API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}

print("[INFO] searching Bing API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()


results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total results for '{}'".format(estNumResults, term))

total = 0

for offset in range(0, estNumResults, GROUP_SIZE):
    
    print("[INFO] making request for group {}-{} of {}...".format(
        offset, offset + GROUP_SIZE, estNumResults))
    params["offset"] = offset
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    print("[INFO] saving images for group {}-{} of {}...".format(
        offset, offset + GROUP_SIZE, estNumResults))

    for v in results["value"]:
        try:
            print("[INFO] fetching: {}".format(v["contentUrl"]))
            r = requests.get(v["contentUrl"], timeout=30)
            # build the path to the output image
            ext = v["contentUrl"][v["contentUrl"].rfind("."):]
            p = os.path.sep.join(
                [args["output"], "{}{}".format(str(total).zfill(8), ext)])
            f = open(p, "wb")
            f.write(r.content)
            f.close()
        except Exception as e:
            if type(e) in EXCEPTIONS:
                print("[INFO] skipping: {}".format(v["contentUrl"]))
                continue

            image = cv2.imread(p)
            if image is None:
                print("[INFO] deleting: {}".format(p))
                os.remove(p)
                continue
            total += 1

