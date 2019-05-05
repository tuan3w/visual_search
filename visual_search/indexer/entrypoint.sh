#!/bin/bash

echo "Running elasticseach instance's healthcheck..."
curl http://elasticsearch:9200/_cat/health

# Set index format of elasticsearch service
echo "Preparing elasticsearh index..."
curl -XPUT http://elasticsearch:9200/im_data -d @schema_es.json

# And index some image data
echo "Indexing images..."
python indexer.py --input ./images --es_host http://elasticsearch --es_port 9200
