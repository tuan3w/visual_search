import cv2
import re
import numpy as np

def read_img_base64(base64str):
    """Read image from base64 string
    """
    #check if contains javascript
    image_data = re.sub('^data:image/.+;base64,', '', base64str)\
        .decode('base64')

    nparr = np.fromstring(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)\
        .astype(np.float32)
    return img

def read_img_blob(im_str):
    """Read image from base64 string
    """
    #check if contains javascript
    # arr = np.asarray(bytearray(im_str), dtype=np.uint8)
    arr = np.fromstring(im_str, np.uint8)
    # img = cv2.imdecode(arr,-1) # 'load it as it is'
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)\
        .astype(np.float32)
    return img
