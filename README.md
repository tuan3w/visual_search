A visual search engine based on Elasticsearch and Tensorflow

![Visual search enging](screenshot.png)
## Requirement
 * Elasticsearch 5.x
 * Tensorflow 1.12.1
 * Flask
 * opencv2

## Setup
 * Elasticsearch:
 The easiest way to setup is using [Docker](https://www.docker.com/) with [Docker Compose](https://docs.docker.com/compose/)
 With `docker-compose` everything you have to done is so simple:
 ```bash
 cd visual_search/elasticsearch
 docker-compose up -d
 ```
 * Build elasticsearch plugin
 We need to build Elasticsearch plugin to compute distance between feature vectors.
 Make sure that you have [Maven](https://maven.apache.org/) installed.

 ```bash
 cd visual_search/es_plugin
 mvn install

 cd target/release
 // create simple server to serve plugin
 python -m 'SimpleHTTPServer' &

 //install plugin
 cd ../elasticsearch
 docker exec -it elasticsearch_elasticsearch_1 elasticsearch-plugin install http://localhost:8000/esplugin-0.0.1.zip
 docker-compose restart
 ```
 * Prepare index
 ```bash
 curl -XPUT http://localhost:9200/img_data -d @schema_es.json
 ```
 * Install r-cnn
 I use `faster r-cnn` implemented by [@Endernewton](https://github.com/endernewton) for object detection
 Follow the [link](https://github.com/endernewton/tf-faster-rcnn) for the guide to setup and download VGG16 model
## Index image to elasticsearch
 ```bash
 export WEIGHT_PATH=...
 export MODEL_PATH=...
 export INPUT=..
 python tool/index_es.py --weight $WEIGHT_PATH --model_path $MODEL_PATH --input $INPUT
 ```
## Start server
 Run:
 ```bash
 python tool/server.py
 ```
 and access to `http://localhost:5000/static/index.html` to test search engine.

 Have fun =))
