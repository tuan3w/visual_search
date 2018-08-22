A visual search engine based on Elasticsearch and Tensorflow

![Visual search enging](screenshot.png)
## Install equirements

```bash
$ cd visual_search
$ pip install -r requirements.txt
```
## Setup
 * Setup Elasticsearch

 The easiest way to setup is using [Docker](https://www.docker.com/) with [Docker Compose](https://docs.docker.com/compose/). With `docker-compose` everything you have to do is so simple:

 ```bash
 cd visual_search/elasticsearch
 docker-compose up -d
 ```

 * Install elasticsearch plugin

 We need to build Elasticsearch plugin to compute distance between feature vectors.
 Make sure that you have [Maven](https://maven.apache.org/) installed.

 ```bash
 $ cd visual_search/es-plugin
 $ mvn install

 $ cd target/release
 // create simple server to serve plugin
 $ python -m 'SimpleHTTPServer' &

 //install plugin
 //go back to elasticsearch folder
 $ cd ../../..
 $ docker exec -it elasticsearch_elasticsearch_1 elasticsearch-plugin install http://localhost:8000/esplugin-0.0.1.zip
 $ docker-compose restart
 ```

 * Index preparation

 ```bash
 curl -XPUT http://localhost:9200/im_data -d @schema_es.json
 ```
 * Setup faster r-cnn

 I used earlier  `faster r-cnn` version implemented by [@Endernewton](https://github.com/endernewton) for object detection. To get pretrained model, please visit [release section](https://github.com/tuan3w/visual_search/releases) and download `model.tar.gz` file, and extracts this file to `visual_search/models` folder.

 You also need to build faster r-cnn library by running following commands:
```bash
$ cd visual_search/lib
$ make
```
## Indexing images to elasticsearch
To index data, just run command:

```bash 
$ python index_es.py --input [image dir]
```
For full comamnd options, please run:
```bash
$ python index_es.py --help
```

## Start server

 Before starting the server, you must to update `IMGS_PATH` variable in `visual_search/server.py` to the location of folder where images are stored.

 ```bash
 cd visual_search
 python server.py
 ```

 and access the link `http://localhost:5000/static/index.html` to test the search engine.

## LICENSE
[MIT](LICENSE)
