# Face Recognition on Video
## A simple yet effective pipeline for face recognition on videos

#### Demo
#### Usage
In  order to use this package you need to follow these steps one by one:
1. Make a conda env 
2. Install dependencies
3. Prepare FaceBoxes
4. Download the face embedding generator's pretrained model
5. Extract face crops and align them using the following command:

```
python extract_faces.py --gallery-root-path /mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/DS_Whole_IMGs --root-path-to-save-crops /mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/CROPPED_FACES
```

6. Extract embeddings
```
 python --checkpoint-to-magface /mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/pretrained_models/magface_epoch_00025.pth --root-path-to-save-embs /mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/EMBS  --root-path-facecrops /mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/CROPPED_FACES
```

7. Train an ID Classifier
```
python SVM_on_embs.py --root-path-to-embs /mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/EMBS  --root-path-to-save-svm-model /mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/pretrained_models/id_classifier.sav --root-path-to-save-label-encoder /mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/pretrained_models/label_encoder.pickle
```