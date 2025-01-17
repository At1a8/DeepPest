#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
#cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.png')
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=1,
            show_class=True,
            thresh=0.8,
            kp_thresh=2
        )

def feature_vis():
    print (workspace.Blobs())
    data = 'gpu_0/res2_0_branch2c'
    for blob in (workspace.Blobs()):
        print (blob)
        print (workspace.FetchBlob(blob).shape)
        print ('feature/' + str(blob)+ '.png')
        if blob == data:
            print (workspace.FetchBlob(data).shape)
            l = len(range(workspace.FetchBlob(data).shape[1]))
            n = int(np.ceil(np.sqrt(l)))
            for xindex in range(n):
                for yindex in range(n):
                    if yindex == 0:
                        im = (workspace.FetchBlob(blob)[0][n*xindex])
                        print (n*xindex+yindex)
                        plt.figure()  
                        plt.imshow(im)  
                        plt.axis('off')
                        plt.savefig('feature/' + str(blob)+ str(n*xindex+yindex)+ '.png', dpi = 100, bbox_inches = "tight")
                    else:
                        #im = np.concatenate([im, workspace.FetchBlob(blob)[0][n*xindex+yindex]], axis=1)
                        im = (workspace.FetchBlob(blob)[0][n*xindex+yindex])
                        print (n*xindex+yindex)
                        plt.figure()  
                        plt.imshow(im)  
                        plt.axis('off')
                        plt.savefig('feature/' + str(blob)+ str(n*xindex+yindex)+ '.png', dpi = 100, bbox_inches = "tight")
                #if xindex == 0:
                    #image = im
                #else:
                    #image = np.vstack((image, im))

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
    #feature_vis()
            #plt.figure()  
            #plt.imshow(image)  
            #plt.axis('off')
            #plt.savefig('feature/' + str(blob)+ '.png', dpi = 100, bbox_inches = "tight")

            #cv2.imwrite('feature/' + str(blob)+ '.jpg', image)
    #cv2.imwrite('feature/' + str(blob)+ '.jpg', workspace.FetchBlob(blob)[0][0])
    #print (workspace.FetchBlob('gpu_0/res3_3_branch2c_bn').shape)
    #cv2.imwrite('0.jpg', workspace.FetchBlob('gpu_0/res3_3_branch2c_bn')[0][0])
