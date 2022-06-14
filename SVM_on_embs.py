from sklearn.svm import SVC
import numpy as np
import argparse 
import pickle
from glob import glob
import os
from sklearn.preprocessing import LabelEncoder
from numpy import genfromtxt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

arg_parser = argparse.ArgumentParser(description='SVM on Embeddings')

arg_parser.add_argument(
    '--root-path-to-embs',  
    type=str,
    required=True,
    help='Abs path to root of the embeddings')

arg_parser.add_argument(
    '--root-path-to-save-svm-model',  
    type=str,
    required=True,
    help='Abs path to save SVM trained model - format: .sav'
    )


arg_parser.add_argument(
    '--root-path-to-save-label-encoder',  
    type=str,
    required=True,
    help='Abs path to save label encoder - format: .pickle'
    )


# arg_parser.add_argument(
#     '--do-assess',  
#     type=bool,
#     required=True,
#     default=True,
#     help='Do you need to check the performance of the ID classifier?'
#     )

args = arg_parser.parse_args()


label_encoder = LabelEncoder()
classifier = SVC(gamma='auto')

if __name__ == "__main__":
    embs_path = [file for file in sorted(glob(os.path.join(args.root_path_to_embs, "*", "*")))]
    
    id_classes = list(set([p.split("/")[-2] for p in embs_path]))
    label_encoder.fit_transform(id_classes)
    # print(list(label_encoder.inverse_transform([2,0,1])))


    data = []
    labels = []
    for e in tqdm(embs_path):
        emb = genfromtxt(e, delimiter=',')
        id_name = e.split("/")[-2]
        data.append(emb)
        labels.append(label_encoder.transform([id_name]).item())


    data = np.array(data)
    labels = np.array(labels)

    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, stratify=labels)
    X_train = data
    y_train = labels
    classifier.fit(X_train, y_train)
    # score = classifier.score(X_test, y_test)


    pickle.dump(classifier, open(args.root_path_to_save_svm_model, 'wb'))
    pickle.dump(label_encoder, open(args.root_path_to_save_label_encoder, 'wb'))

    # print(score)

    
    





