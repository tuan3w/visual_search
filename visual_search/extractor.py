import _init_paths
from model.test import extract_regions_and_feats \
    as _extract_regions_and_feats
from model.test import extract_imfea \
    as _extract_imfea
from math import ceil
import numpy as np
import cv2

import tensorflow as tf
from nets.vgg16 import vgg16
from utils.im_util import read_img_base64

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# logger = logging.getLogger(__name__)


def _binarize_fea(x, thresh):
    '''binary and pack feature vector'''
    binary_vec = np.where(x >= thresh, 1, 0)
    f_len = binary_vec.shape[0]
    if f_len % 32 != 0:
        new_size = int(ceil(f_len / 32.) * 32)
        num_pad = new_size - f_len
        binary_vec = np.pad(binary_vec, (num_pad, 0), 'constant')

    return np.packbits(binary_vec).view('uint32')


class Extractor:
    """ Feature extractor

    Parameter:
        - model_path: path to the model
        - weight_path: weight path
        - sess: tensorflow session
        - num_classes: number of classes
    """
    def __init__(self, model_path, weight_path, sess=None,
                 anchors=[4, 8, 16, 32], num_classes=81):
        if not sess:
            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.allow_growth = True
            # init session
            sess = tf.Session(config=tfconfig)
        # load network
        self.sess = sess
        self.net = vgg16(batch_size=1)
        self.num_classes = num_classes
        # load model
        self.net.create_architecture(sess, "TEST", num_classes, weight_path,
                                     tag='default', anchor_scales=anchors)

        print ('Loading model check point from {:s}').format(model_path)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

    def extract_regions_and_feats(self, img):
        """Extract regions and feature corresponding to
        the each box"""
        boxes = _extract_regions_and_feats(self.sess,
                                           self.net, img, max_per_image=10,
                                           max_per_class=3, thresh=0.1)
        return boxes

    def extract_imfea(self, img):
        "Extract feature for image"
        if type(img) == str:
            img = self.read_img(img)

        fea = _extract_imfea(self.sess, self.net, img)
        return fea

    def binarize_fea(self, fea, thres=0.1):
        "Binarize and pack feature vector"
        return _binarize_fea(fea, thres)

    def get_tags(self, img):
        boxes = self.extract_regions_and_feats(img)
        out = {}
        for cl, bb in boxes.iteritems():
            best_score = max([b['score'] for b in bb])
            out[cl] = float(best_score)
        return out

    def read_img(self, path):
        " Read image from file "
        im = cv2.imread(path)\
            .astype(np.float32, copy=True)
        return im


if __name__ == '__main__':
    model_path = 'output/vgg16/coco_2014_train+coco_2014_valminusminival/'\
        'default/vgg16_faster_rcnn_iter_490000.ckpt'
    weight_path = './data/imagenet_weights/vgg16.weights'

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    # init session
    sess = tf.Session(config=tfconfig)
    extractor = Extractor(model_path, weight_path, sess=sess)

    test_img_path = 'tumblr/tumblr_o56polZl2L1v0xzvvo1_500.jpg'
    with open('test.txt') as f:
        text = f.read().strip()

    img = read_img_base64(text)
    extractor.get_tags(img)
    import pdb; pdb.set_trace()
    # fea = extractor.extract_imfea(img)
    print(fea.shape)
    bin_fea = extractor.binarize_fea(fea)
    print(bin_fea.shape)
    sess.close()
