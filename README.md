A visual search engine based on Elasticsearch and Tensorflow
**TODO**: Fix this problem...
```
Attaching to vs_es_container, vs_container
vs_container     | Warning: Couldn't read data from file "schema_es.json", this makes an empty
vs_container     | Warning: POST.
vs_es_container  | [2019-05-03T09:35:59,663][INFO ][o.e.n.Node               ] [] initializing ...
vs_container     |   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
vs_es_container  | [2019-05-03T09:35:59,716][INFO ][o.e.e.NodeEnvironment    ] [JOUYYv2] using [1] data paths, mounts [[/usr/share/elasticsearch/data (/dev/mapper/ubuntu--vg-root)]], net usable_space [664.9gb], net total_space [914.7gb], spins? [possibly], types [ext4]
vs_container     |                                  Dload  Upload   Total   Spent    Left  Speed
vs_es_container  | [2019-05-03T09:35:59,717][INFO ][o.e.e.NodeEnvironment    ] [JOUYYv2] heap size [1.9gb], compressed ordinary object pointers [true]
vs_es_container  | [2019-05-03T09:35:59,719][INFO ][o.e.n.Node               ] node name [JOUYYv2] derived from node ID [JOUYYv2_Q-SL3pQNb5rIuw]; set [node.name] to override
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0curl: (7) Failed to connect to elasticsearch port 80: Connection refused
vs_es_container  | [2019-05-03T09:35:59,719][INFO ][o.e.n.Node               ] version[5.3.1], pid[1], build[5f9cf58/2017-04-17T15:52:53.846Z], OS[Linux/4.15.0-47-generic/amd64], JVM[Oracle Corporation/OpenJDK 64-Bit Server VM/1.8.0_121/25.121-b13]

```
![Visual search enging](screenshot.png)
## Install requirements

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

 I used earlier  `faster r-cnn` version implemented by [@Endernewton](https://github.com/endernewton) for object detection. To get pretrained model, please visit [release section](https://github.com/tuan3w/visual_search/releases), download and extract file `model.tar.gz` to `visual_search/models` folder.

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

 Now, you can access the link `http://localhost:5000/static/index.html` to test the search engine.

## LICENSE
[MIT](LICENSE)
