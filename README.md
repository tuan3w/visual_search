A visual search engine based on Elasticsearch and Tensorflow

![Visual search enging](screenshot.png)
## Requirements
 There are serveral python libraries you must to install before building the search engine.

 * `elasticsearch==5.2.0`
 * `Tensorflow==1.12.1`
 * `Flask`
 * `opencv-python`

## Setup
 * Setup Elasticsearch

 The easiest way to setup is using [Docker](https://www.docker.com/) with [Docker Compose](https://docs.docker.com/compose/). With `docker-compose` everything you have to do is so simple:

 ```bash
 cd visual_search/elasticsearch
 docker-compose up -d
 ```

 * Building elasticsearch plugin

 We need to build Elasticsearch plugin to compute distance between feature vectors.
 Make sure that you have [Maven](https://maven.apache.org/) installed.

 ```bash
 cd visual_search/es-plugin
 mvn install

 cd target/release
 // create simple server to serve plugin
 python -m 'SimpleHTTPServer' &

 //install plugin
 cd ../elasticsearch
 docker exec -it elasticsearch_elasticsearch_1 elasticsearch-plugin install http://localhost:8000/esplugin-0.0.1.zip
 docker-compose restart
 ```

 * Index preparation

 ```bash
 curl -XPUT http://localhost:9200/img_data -d @schema_es.json
 ```
 * Setup faster r-cnn

 I use `faster r-cnn` version implemented by [@Endernewton](https://github.com/endernewton) for object detection. Follow the [link](https://github.com/endernewton/tf-faster-rcnn) for the guide how to setup faster r-cnn and download VGG16 model trained on COCO dataset.
## Indexing images to elasticsearch

 ```bash
 export WEIGHT_PATH=...
 export MODEL_PATH=...
 export INPUT=..
 cd visual_search
 python index_es.py --weight $WEIGHT_PATH --model_path $MODEL_PATH --input $INPUT
 ```
## Start server

 Before starting the server, you must to update `IMGS_PATH` variable in `visual_search/server.py` to the location of folder where images are stored.

 ```bash
 cd visual_search
 python server.py
 ```

 and access the link `http://localhost:5000/static/index.html` to test the search engine.

 Have fun =))
