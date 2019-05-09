import os
import argparse
import json
from types import SimpleNamespace as Namespace

import cv2
import pandas as pd
import numpy as np

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from logging_config import logger


def parse_args():
    parser = argparse.ArgumentParser(description='Saved results formatter.')
    parser.add_argument('-rod', '--root_output_dir', type=str, required=True)
    parser.add_argument('-dc', '--dataset_config', type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
    config_file = "../configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml"
    config_file = "../configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    # cfg.MODEL.KEYPOINT_ON = True
    # cfg.MODEL.MASK_ON = False

    coco_demo = COCODemo(
        cfg,
        min_image_size=256,
        confidence_threshold=0.7,
        # show_mask_heatmaps=True
    )
    # load image and then run prediction

    with open(args.dataset_config, 'r') as fin:
        # Load config into nested Namespace object
        config = json.load(fin, object_hook=lambda d: Namespace(**d))
    record_csv = pd.read_csv(config.data_loader.args.dataset_args.label_csv)
    rgb_root = config.data_loader.args.dataset_args.rgb_root

    for idex, entry in record_csv.iterrows():
        participant_id, video_id = entry['participant_id'], entry['video_id']
        logger.info(f"Doing video {participant_id}_{video_id}")
        start_frame, stop_frame = int(entry['start_frame']), int(entry['stop_frame'])
        # For debugging
        if start_frame < 3000 and False:
            continue
        video_dir = os.path.join(rgb_root, participant_id, video_id)
        output_dir = os.path.join(args.root_output_dir, participant_id, video_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        batch_size = 64
        for i in range(start_frame, stop_frame, batch_size):
            if i == start_frame:
                logger.info(f"Frame {i}")
            images = []
            for j in range(i, i + batch_size):
                if j >= stop_frame:
                    break
                image = cv2.imread(os.path.join(video_dir, f'frame_{j:010d}.jpg'))
                images.append(image)
            predictions = coco_demo.compute_predictions(images, do_mask=True)
            for j, prediction in enumerate(predictions):
                top_predictions = coco_demo.select_top_predictions(prediction)
                # Humna only
                indices = [i for i, label in enumerate(top_predictions.get_field('labels')) if label == 1]
                top_predictions = top_predictions[indices]
                seg = np.zeros(images[0].shape, dtype=np.float32)
                seg = coco_demo.overlay_mask(seg, top_predictions, cv2.FILLED)
                output_path = os.path.join(output_dir, f'seg_{i+j:010d}.png')
                cv2.imwrite(output_path, seg)
                """
                scores = top_predictions.get_field("scores").tolist()
                labels = top_predictions.get_field("labels").tolist()
                boxes = top_predictions.bbox
                results = []
                for box, score, label in zip(boxes, scores, labels):
                    if label == 1:
                        results.append(box.numpy().tolist() + [score])
                # if len(results) > 0:
                #    logger.info(results)

                output_path = os.path.join(output_dir, f'bbox_{i+j:010d}.txt')
                np.savetxt(output_path, np.array(results))

                """


if __name__ == "__main__":
    args = parse_args()
    main(args)
