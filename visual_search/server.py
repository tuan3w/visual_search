#!/usr/bin/env python


from flask import Flask
from flask import request, jsonify, \
    send_from_directory

import base64
import tensorflow as tf
from elasticsearch import Elasticsearch

from extractor import Extractor
from utils.im_util import read_img_blob
from es.ImFea_pb2 import ImFea, ImFeaArr,\
        ImFeaBinArr, ImFeaBin


IMGS_PATH = './tumblr/'
app = Flask(__name__, static_url_path='')

class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def load_model():
    """Load feature extractor model"""

    # create feature extractor
    weight_path = './models/vgg16.weights'
    model_path = './models/faster_rcnn_models/' \
        'vgg16_faster_rcnn_iter_490000.ckpt'

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    # init session
    sess = tf.Session(config=tfconfig)
    extractor = Extractor(model_path, weight_path, sess=sess)
    return extractor

extractor = load_model()
es = Elasticsearch(hosts='localhost:9200')

@app.route("/hello", methods=['GET'])
def hello():
    return "Hello, world!"


@app.route("/extract_fea", methods=['GET', 'POST'])
def extract_fea():
    imgStr = request.values.get('img')
    if imgStr is None:
        raise InvalidUsage('parameter "img" is missing', status_code=410)
    try:
        img = read_img_blob(imgStr)
    except:
        raise InvalidUsage('Invalid "img" param, must be a base64 string',
                           status_code=410)
    fea = extractor.extract_imfea(img)
    is_binary = request.values.get('is_binary')
    if is_binary and is_binary == 'true':
        fea = extractor.binarize_fea(fea)
        fea_obj = ImFeaBin()
    else:
        fea_obj = ImFea()
    fea_obj.f.extend(fea)
    base64str = base64.b64encode(fea_obj.SerializeToString())

    out = {}
    out['fea'] = base64str
    return jsonify(out)


@app.route("/get_tags", methods=['GET', 'POST'])
def get_tags():
    """get tags corresponding to a image"""
    if not 'img' in request.files:
        raise InvalidUsage('parameter "img" is missing', status_code=410)
    try:
        f = request.files.get('img')
        img_str = f.read()
        img = read_img_blob(img_str)
    except:
        raise InvalidUsage('Invalid "img" param, must be a blob string',
                           status_code=410)
    tags = extractor.get_tags(img)
    out = {}
    out['tags'] = tags
    return jsonify(out)



QUERY = """
{
"_source": ["im_src", "cl", "coords"],
"query": {
  "function_score" : {
    "query" : {
      "match_all" : {
        "boost" : 1.0
      }
    },
    "functions" : [
      {
        "filter" : {
          "match_all" : {
            "boost" : 1.0
          }
        },
        "script_score" : {
          "script" : {
            "inline" : "hamming_score",
            "lang" : "native",
            "params" : {
              "f" : "bin_sigs",
              "fea" : [##fea##],
              "verbose" : true
            }
          }
        }
      }
    ],
    "score_mode" : "sum",
    "boost_mode" : "replace",
    "max_boost" : 3.4028235E38,
    "boost" : 1.0
  }
}
}
"""
@app.route("/search", methods=['GET', 'POST'])
def search():
    """get tags corresponding to a image"""
    if not 'img' in request.files:
        raise InvalidUsage('parameter "img" is missing', status_code=410)
    try:
        f = request.files.get('img')
        img_str = f.read()
        img = read_img_blob(img_str)
    except:
        raise InvalidUsage('Invalid "img" param, must be a blob string',
                           status_code=410)
    fea = extractor.extract_imfea(img)
    fea = extractor.binarize_fea(fea)
    fea_str = ','.join([str(int(t)) for  t in fea])
    query = QUERY.replace('##fea##', fea_str)
    print(query)
    result = es.search(index='im_data', doc_type='obj', body=query)
    rs = []
    if 'hits' in result and \
        'hits' in result['hits']:
        #distinct
        all_imgs = set([])
        hits = result['hits']['hits']
        for hit in hits:
            o = hit['_source']
            o['score'] = hit['_score']
            #update im_src
            im_src = '/img/{}'.format(o['im_src'])
            if not im_src in all_imgs:
                o['im_src'] = im_src
                all_imgs.add(im_src)
                rs.append(o)
        print all_imgs

    out = {}
    out['hits'] = rs
    return jsonify(out)


@app.route('/static/<path:path>')
def send_static_files(path):
    "static files"
    return send_from_directory('static_data', path)


@app.route('/img/<path:path>')
def send_image(path):
    "static files"
    return send_from_directory(IMGS_PATH, path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
