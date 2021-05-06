import os
import requests
from time import time
from multiprocessing.pool import ThreadPool

def url_response(url):
  path, url = url
  r = requests.get(url, stream = True)
  print(path)
  with open(path, 'wb') as f:
    for chunk in r.iter_content(chunk_size = 1024):
      if chunk:
        f.write(chunk)
urls = [("train_32x32.mat",'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'),
("test_32x32.mat",'http://ufldl.stanford.edu/housenumbers/test_32x32.mat')]

for x in urls:
  url_response(x)
