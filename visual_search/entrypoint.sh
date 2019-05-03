#!/bin/bash

# Set index format of elasticsearch service
curl -XPUT http://elasticsearch/im_data -d @schema_es.json

# And index some image data
python index_es.py --input ./tumblr --es_host http://elasticsearch

# Start the application server
python server.py
