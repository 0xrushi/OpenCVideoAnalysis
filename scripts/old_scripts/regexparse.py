'''
This script uses regex to download dropbox links.
Since we don't have any dropbox-cli downloader 
for public links, and we cannot use plain web scraping
to scrape dynamic pages of dropbox
'''

import re
with open("scrape.html", "r") as f:
    st = f.read()

# print(type(st))
pp = re.findall(r'(?<=href\=\")[\w\:\/\.\?\=\-]+dl\=0(?=")', st)
print(len(pp))

with open("dropboxlinks.txt", "w") as f:
    for i in pp:
        f.write(i+'\n')