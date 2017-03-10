# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------

import cv2
import numpy as np
import cPickle
import os
import math

from utils.timer import Timer
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in xrange(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  # seems to have height, width, and image scales
  # still not sure about the scale, maybe full image it is 1.
  blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  _, scores, bbox_pred, rois, feats = net.test_image(sess, blobs['data'], blobs['im_info'])

  boxes = rois[:, 1:5] / im_scales[0]

  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes, feats

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(num_classes)]
  for cls_ind in xrange(num_classes):
    for im_ind in xrange(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(dets, thresh)
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.05):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in xrange(num_images)]
         for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  for i in xrange(num_images):
    p = 'data/coco/images/val2014/COCO_val2014_000000532481.jpg'
    # im = cv2.imread(imdb.image_path_at(i))
    im = cv2.imread(p)

    _t['im_detect'].tic()
    scores, boxes, feats = im_detect(sess, net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    # skip j = 0, because it's the background class
    for j in xrange(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in xrange(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in xrange(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
    _t['misc'].toc()

    print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time,
            _t['misc'].average_time)


  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

  print 'Evaluating detections'
  imdb.evaluate_detections(all_boxes, output_dir)


classes= ('__background__', u'person', u'bicycle', u'car', u'motorcycle', u'airplane', u'bus', u'train', u'truck', u'boat', u'traffic light', u'fire hydrant', u'stop sign', u'parking meter', u'bench', u'bird', u'cat', u'dog', u'horse', u'sheep', u'cow', u'elephant', u'bear', u'zebra', u'giraffe', u'backpack', u'umbrella', u'handbag', u'tie', u'suitcase', u'frisbee', u'skis', u'snowboard', u'sports ball', u'kite', u'baseball bat', u'baseball glove', u'skateboard', u'surfboard', u'tennis racket', u'bottle', u'wine glass', u'cup', u'fork', u'knife', u'spoon', u'bowl', u'banana', u'apple', u'sandwich', u'orange', u'broccoli', u'carrot', u'hot dog', u'pizza', u'donut', u'cake', u'chair', u'couch', u'potted plant', u'bed', u'dining table', u'toilet', u'tv', u'laptop', u'mouse', u'remote', u'keyboard', u'cell phone', u'microwave', u'oven', u'toaster', u'sink', u'refrigerator', u'book', u'clock', u'vase', u'scissors', u'teddy bear', u'hair drier', u'toothbrush')

def get_boxes(sess, net, img, all_feats, all_boxes):
  boxes = {}
  for cls_ind, cls in enumerate(classes):
    if cls == '__background__':
      continue
    dets = all_boxes[cls_ind]
    feats = all_feats[cls_ind]
    if dets == []:
      continue
    for k in xrange(dets.shape[0]):
      det = dets[k]
      box = {}
      box['lt'] = [int(det[0]), int(det[1])]
      box['rb'] = [int(det[2]), int(det[3])]
      box['f'] = feats[k]
      box['cl'] = cls
      box['score'] = det[4]
      box_width = box['rb'][0] - box['lt'][0]
      box_height = box['rb'][1] - box['lt'][1]
      min_size = min(box_width, box_height)
      if min_size > 10:
        if not cls in boxes:
           boxes[cls] = []
        class_boxes = boxes[cls]
        # boxes.append(box)
        class_boxes.append(box)
  return boxes



def extract_regions_and_feats(sess, net, im, max_per_image=10, max_per_class=3, thresh=0.1):
  """Extract regions and features with respect to each region"""
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
 # timecrs
  _t = {'im_detect' : Timer(), 'misc' : Timer()}
  if type(im) == str:
      im = cv2.imread(im)

  _t['im_detect'].tic()
  scores, boxes, feats = im_detect(sess, net, im)
  _t['im_detect'].toc()

  _t['misc'].tic()

  all_boxes = [[] for _ in xrange(81)]
  all_feats = [[] for _ in xrange(81)]
  # skip j = 0, because it's the background class
  for j in xrange(1, 81):
    image_thresh = np.sort(scores[:,j])[-max_per_image]
    th = thresh if thresh > image_thresh else image_thresh
    inds = np.where(scores[:, j] > th)[0]
    cls_scores = scores[inds, j]
    cls_boxes = boxes[inds, j*4:(j+1)*4]
    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(cls_dets, cfg.TEST.NMS)
    cls_dets = cls_dets[keep, :]
    feats_part = feats[keep, :]
    all_boxes[j] = cls_dets
    all_feats[j] = feats_part

  # Limit to max_per_image detections *over all classes*
  if max_per_image > 0:
    image_scores = np.hstack([all_boxes[j][:, -1]
        for j in xrange(1, 81)])
    if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in xrange(1, 81):
            keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
            all_boxes[j] = all_boxes[j][keep, :]
            all_feats[j] = all_feats[j][keep, :]
  _t['misc'].toc()

  # print 'im_detect in {:.3f}s {:.3f}s' \
          # .format(_t['im_detect'].average_time,
                  # _t['misc'].average_time)

  boxes = get_boxes(sess, net, im, all_feats, all_boxes)
  return boxes

def extract_imfea(sess, net, img):
  # im = cv2.imread(path)
  # im_orig = im.astype(np.float32, copy=True)
  # resized image first
  resized_im = cv2.resize(img, (224, 224))
  resized_im -= cfg.PIXEL_MEANS
  fea = net.extract_fc7(sess, [resized_im])
  return np.squeeze(fea)

