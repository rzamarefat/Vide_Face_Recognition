# from FaceDetector import FaceDetector 

import argparse
from glob import glob
import os
from uuid import uuid1
from skimage import transform as trans
from mtcnn import MTCNN
import numpy as np
import cv2


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Face Extraction Pipeline')


    arg_parser.add_argument(
        '--gallery-root-path',  
        type=str,
        required=True,
        help='Abs path to Gallery of Individuals')

    arg_parser.add_argument(
        '--root-path-to-save-crops',  
        type=str,
        required=True,
        help='Abs path to save face crops')
    
    args = arg_parser.parse_args()

    people_in_gallery = [file.split("/")[-2] for file in glob(os.path.join(args.gallery_root_path, "*", "*"))]

    arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)
    

    if not(os.path.isdir(args.root_path_to_save_crops)):
        os.mkdir(args.root_path_to_save_crops)
    
    for p in people_in_gallery:
        if not(os.path.isdir(os.path.join(args.root_path_to_save_crops, p))):
            os.mkdir(os.path.join(args.root_path_to_save_crops, p))

    # face_detector = FaceDetector()
    detector = MTCNN()
    tform = trans.SimilarityTransform()
    for file in sorted(glob(os.path.join(args.gallery_root_path, "*", "*"))):
        person = file.split("/")[-2]

        img  = cv2.imread(file)
        for box in detector.detect_faces(img):
            # if box["confidence"] > 0.98:
            unique_id_name = str(uuid1())
            path_to_save_aligned = os.path.join(args.root_path_to_save_crops, person, f"{person}__{unique_id_name}__ALIGNED.jpg")
            path_to_save_not_aligned = os.path.join(args.root_path_to_save_crops, person, f"{person}__{unique_id_name}__NOTALIGNED.jpg")

            ldmks = np.array([box['keypoints']['left_eye'], box['keypoints']['right_eye'], box['keypoints']['nose'], box['keypoints']['mouth_left'], box['keypoints']['mouth_right']])
            


            x,y,w,h = box['box'][0], box['box'][1], box['box'][2], box['box'][3]

            cut_no_align = img[y:y+h, x:x+w]
            cut_no_align = cv2.resize(cut_no_align, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(path_to_save_not_aligned, cut_no_align)

            tform.estimate(ldmks, arcface_src)
            M = tform.params[0:2, :]
            cut = cv2.warpAffine(img, M, (112, 112))
            cv2.imwrite(path_to_save_aligned, cut)