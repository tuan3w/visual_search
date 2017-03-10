import _init_paths

import cv2
import glob
import numpy as np
import argparse
import tensorflow as tf

from base64 import b64encode

from elasticsearch import Elasticsearch
from elasticsearch import helpers

from extractor import Extractor
from es.ImFea_pb2 import ImFea, ImFeaArr, \
    ImFeaBinArr, ImFeaBin

import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_PATH = './models/faster_rcnn_models/'\
    'vgg16_faster_rcnn_iter_490000.ckpt'
WEIGHT_PATH = './models/vgg16.weights'


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Index image to elasticsearch')
    parser.add_argument('--weight', dest='weight',
                        help='weight to test',
                        default=WEIGHT_PATH, type=str)
    parser.add_argument('--model_path', dest='model_path',
                        help='path to the model',
                        default=MODEL_PATH, type=str)

    parser.add_argument('--input', dest='input',
                        help='Input image folder',
                        default=None, type=str)

    parser.add_argument('--es_host', dest='es_host',
                        help='es sever host',
                        default='localhost', type=str)
    parser.add_argument('--es_index', dest='es_index',
                        help='index name',
                        default='im_data', type=str)
    parser.add_argument('--es_type', dest='es_type',
                        help='index type',
                        default='obj', type=str)
    parser.add_argument('--es_port', dest='es_port',
                        help='es server port',
                        default=9200, type=int)

    args = parser.parse_args()

    if not args.input:
        parser.error('Input folder not given')
    return args


def create_doc(im_src, tag, coords, fea_arr, fea_bin_arr):
    """
    Create elasticsearch doc

    Params:
        im_src: image file name
        tag: tag or class for image
        coords: list of boxes corresponding to a tag
        fea_arr: list of ImFea objects
        fea_bin_arr: list of ImFeaBin objects
    """
    doc = {}
    doc['coords'] = coords
    f_bin = ImFeaBinArr()
    f = ImFeaArr()
    f.arr.extend(fea_arr)
    f_bin.arr.extend(fea_bin_arr)
    obj_bin_str = b64encode(f_bin.SerializeToString())
    obj_str = b64encode(f.SerializeToString())
    doc['sigs'] = obj_str
    doc['bin_sigs'] = obj_bin_str
    doc['im_src'] = im_name
    doc['cl'] = tag
    return doc


if __name__ == '__main__':
    args = parse_args()

    # init session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    # create feature extractor
    print args
    extractor = Extractor(args.model_path,
                          args.weight,
                          sess=sess)

    # create elasticsearch client
    es = Elasticsearch(hosts='{}:{}'.format(args.es_host, args.es_port))

    # load images
    images = glob.glob(args.input + "/*")

    bulk = []
    actions = []
    num_docs = 0
    count = 0
    es_index = args.es_index
    es_type = args.es_type
    num_imgs = len(images)
    for im_path in images:
        # read image
        im = cv2.imread(im_path) \
            .astype(np.float32, copy=True)

        boxes = extractor.extract_regions_and_feats(im)

        count += 1
        if count % 100 == 0:
            logger.info('Processing image {}/{}'.format(count, num_imgs))

        for cl, cl_boxes in boxes.iteritems():
            coords = []
            im_name = im_path.split('/')[-1]
            ar = []
            ar_bin = []
            for b in cl_boxes:
                coord_box = {}
                coord_box['c'] = b['lt'] + b['rb']
                coord_box['score'] = float(b['score'])
                coords.append(coord_box)
                f = b['f']
                f_bin = extractor.binarize_fea(f)

                im_fea = ImFea()
                im_fea_bin = ImFeaBin()
                im_fea.f.extend(f)
                im_fea_bin.f.extend(f_bin)
                ar.append(im_fea)
                ar_bin.append(im_fea_bin)

            doc = create_doc(im_name, cl, coords, ar, ar_bin)
            num_docs += 1

            # create index action
            action = {
                "_index": es_index,
                "_type": es_type,
                "_source": doc
            }
            actions.append(action)
            if len(actions) == 1000:
                logger.info('Bulking {} docs to sever, indexed: {}'
                            .format(len(actions), num_docs))
                helpers.bulk(es, actions)
                del actions[:]

        # index document ifself
        im_fea = extractor.extract_imfea(im)
        im_fea_bin = extractor.binarize_fea(im_fea)
        doc = {}
        (w, h, _) = im.shape
        coords = [{'c': [0, 0, h, w], 'score': 1.0}]
        fea_bin = ImFeaBin()
        fea = ImFea()
        fea_bin.f.extend(im_fea_bin)
        fea.f.extend(im_fea)
        doc = create_doc(im_name, 'whole', coords, [fea], [fea_bin])
        num_docs += 1

        # create index action
        action = {
            "_index": es_index,
            "_type": es_type,
            "_source": doc
        }
        actions.append(action)
        if len(actions) == 1000:
            logger.info('Bulking {} docs to sever, indexed {}'
                        .format(len(actions), num_docs))
            helpers.bulk(es, actions)
            del actions[:]

    if len(actions) > 0:
        helpers.bulk(es, actions)
        logger.info('Bulking {} docs to sever,  total {}'
                    .format(len(actions), num_docs))

    sess.close()
