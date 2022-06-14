import torch 
from MagFace.MagFace import builder_inf
from glob import glob
import argparse
import os 
import cv2
from torchvision import transforms
from numpy import savetxt
from tqdm import tqdm

arg_parser = argparse.ArgumentParser(description='Face Embedding Extraction Pipeline')

arg_parser.add_argument(
    '--checkpoint-to-magface',  
    type=str,
    required=True,
    help='Abs path to MagFace Checkpoint')

arg_parser.add_argument(
    '--root-path-to-save-embs',  
    type=str,
    required=True,
    help='Abs root path to store MagFace Embs')

arg_parser.add_argument(
    '--root-path-facecrops',  
    type=str,
    required=True,
    help='Abs root path to read aligned face crops')


args = arg_parser.parse_args()


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112))
])

if __name__ == "__main__":
     
    model = builder_inf(ckpt_path=args.checkpoint_to_magface, cpu_mode=True)

    files = [file for file in sorted(glob(os.path.join(args.root_path_facecrops, "*", "*")))]
    
    for f in tqdm(files):
        img  = cv2.imread(f)
        img = transforms(img)
        img = torch.unsqueeze(img, dim=0)
        data = torch.vstack((img, torch.randn(1, 3, 112, 112)))
        emb = model(data)[0]
        emb = emb.detach().to('cpu').numpy()
        person_name = f.split("/")[-2]

        if not(os.path.isdir(os.path.join(args.root_path_to_save_embs, person_name))):
            os.mkdir(os.path.join(args.root_path_to_save_embs, person_name))
        
        path_to_save_emb = os.path.join(args.root_path_to_save_embs, person_name, f"{f.split('/')[-1].split('.')[0]}.csv")
        savetxt(path_to_save_emb, emb, delimiter=",")
