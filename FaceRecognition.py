import sys
import os

sys.path.append(os.getcwd())


from FaceBoxes.FaceBoxes import FaceBoxes
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from MagFace.MagFace import builder_inf
import pickle
import torch


from uuid import uuid1

class FaceRecognition():
    def __init__(self, 
        face_emb_gen_ckpt_path, 
        face_classifier_trained_model_path,
        label_encoder_path,
        face_margin=0.0, 
        face_crop_dim=(112, 112),
        cpu_mode = False,
        save_result=True
        ):


        self.face_emb_gen_ckpt_path = face_emb_gen_ckpt_path
        self.face_classifier_trained_model_path = face_classifier_trained_model_path
        self.label_encoder_path = label_encoder_path
        self.save_result=save_result
        self.cpu_mode = cpu_mode
        self.faceboxes = FaceBoxes()
        self.face_margin = face_margin
        self.face_crop_dim = face_crop_dim

        self.inference_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            
        ])

        self.face_emb_ckpt_path = "/mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/pretrained_models/magface_epoch_00025.pth"
        self.face_emb_generator = builder_inf(ckpt_path=self.face_emb_ckpt_path, cpu_mode=True)
        # self.face_emb_generator.eval()

        self.face_classifier = pickle.load(open(face_classifier_trained_model_path, 'rb'))

        self.label_encoder = pickle.load(open(self.label_encoder_path, 'rb'))
        print(self.label_encoder.classes_)

    def read_img(self, path):
        return cv2.imread(path)

    def detect_faces(self, frame):    
        face_boxes = FaceBoxes(timer_flag=True)
        dets = face_boxes(frame)  # xmin, ymin, w, h
        return dets

    def apply_img_transformations(self, face):
        return self.inference_transforms(face)

    def crop_faces(self, frame, dets):
        faces = []
        for d in dets:
            (x, y), (x2, y2) = (int(d[0]), int(d[1])), (int(d[2]), int(d[3]))
            width, height = x2 - x, y2 - y
            face = frame[max(y-int(self.face_margin*height), 0):min(y2+int(self.face_margin*height), frame.shape[0]), max(x-int(self.face_margin*width), 0):min(x2+int(self.face_margin*width), frame.shape[1])]

            face = cv2.resize(face, self.face_crop_dim)

            face = Image.fromarray(face)

            face = self.apply_img_transformations(face)
            face = face.permute(1,2,0).detach().to('cpu').numpy()
            face = face * 250
            face = face.astype(np.int32)
            # cv2.imwrite(f"./{str(uuid1())}.jpg", face)
            faces.append(face)

        return faces

    def generate_emb(self, faces):
        if len(faces) == 1:
            print(faces[0].shape)
            face = torch.Tensor(faces[0])
            face = face.permute(2, 0, 1)
            face = torch.unsqueeze(face, dim=0)
            face = torch.vstack((face, torch.randn(1, 3, 112, 112)))
            emb = self.face_emb_generator(face)[0]
            emb = emb.detach().to('cpu').numpy()
            emb = emb[np.newaxis, :]
            return emb

        
        faces = torch.Tensor(faces)
        faces = faces.permute(0, 3, 1, 2)

        embs = self.face_emb_generator(faces)
        embs = embs.detach().to('cpu').numpy()
        return embs

    def recognize_faces(self, faces):

        embs = self.generate_emb(faces)

        classified_faces = self.face_classifier.predict(embs)
        classified_faces = self.label_encoder.inverse_transform(classified_faces)

        return classified_faces

    def draw_bounding_boxes(self, image, dets, classification_data):
        for index, d in enumerate(dets):
            (x, y), (x2, y2) = (int(d[0]), int(d[1])), (int(d[2]), int(d[3]))
            width, height = x2 - x, y2 - y

            start_point = (max(x-int(self.face_margin*width), 0), max(y-int(self.face_margin*height), 0))

            end_point =  (min(x2+int(self.face_margin*width), image.shape[1]), min(y2+int(self.face_margin*height), image.shape[0]))
            
            color = (36,255,12) if classification_data[index] == "UNK" else (255,36,12)

            cv2.rectangle(image, start_point, end_point, color, 2)
            cv2.putText(image, classification_data[index], (x-2, y+height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image


    def run_recognition(
        self,
        path_to_video,
        ):

        # path_to_img = "/mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/sample.jpg"
        # path_to_img = "/mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/sample_one_face.jpg"
        # path_to_img = "/mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/UNK_TRUMP.jpg"

        # frame = cv2.imread(path_to_img)
        
        vid = cv2.VideoCapture(path_to_video)

        if self.save_result:
            frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

        num = 0
        while(True):
            num += 1
            ret, frame = vid.read()
            
            dets = self.detect_faces(frame)

            if not(len(dets) == 0):
                

                faces = self.crop_faces(frame=frame, dets=dets)
                
                recognized_people = self.recognize_faces(faces)

                image = self.draw_bounding_boxes(frame, dets, recognized_people)
                # cv2.imwrite(f"/mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/frames/{num}__{str(uuid1())}.jpg", image)    
        

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            output.write(frame)

        vid.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # import argparse
    # arg_parser = argparse.ArgumentParser(description='Face Embedding Extraction Pipeline')

    # arg_parser.add_argument(
    #     '--checkpoint-to-magface',  
    #     type=str,
    #     required=True,
    #     help='Abs path to MagFace Checkpoint')

    # arg_parser.add_argument(
    #     '--root-path-to-save-embs',  
    #     type=str,
    #     required=True,
    #     help='Abs root path to store MagFace Embs')

    # arg_parser.add_argument(
    #     '--root-path-facecrops',  
    #     type=str,
    #     required=True,
    #     help='Abs root path to read aligned face crops')


    # args = arg_parser.parse_args()

    face_emb_gen_ckpt_path = "/mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/pretrained_models/magface_epoch_00025.pth"
    face_classifier_trained_model_path = "/mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/pretrained_models/id_classifier.sav"
    label_encoder_path = "/mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/pretrained_models/label_encoder.pickle"
    path_to_video = "/mnt/829A20D99A20CB8B/projects/github_projects/VideoFaceRecognition/001_cut.m4v"


    face_recongition = FaceRecognition(
        face_emb_gen_ckpt_path, 
        face_classifier_trained_model_path,
        label_encoder_path,
        face_margin=0.1, 
        face_crop_dim=(112, 112),
        cpu_mode = False,
        save_result=True
    )


    face_recongition.run_recognition(
        path_to_video
    )
    