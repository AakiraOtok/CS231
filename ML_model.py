import cv2
import os
import numpy as np
from scipy.spatial.distance import cdist
import sys
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET
import os
import cv2
import random
import sys
import glob

from math import sqrt
import sklearn.svm as svm
import pickle
from sklearn.metrics import f1_score

Traffic_name2idx = {
    "w.245a" : 1,
    "r.434" : 2,
    "w.202a" : 3,
    "r.301d" : 4,
    "p.103c" : 5,
    "s.509a" : 6,
    "p.130" : 7,
    "w.203c" : 8,
    "r.425" : 9,
    "p.137" : 10,
    "p.127" : 11,
    "w.208" : 12,
    "w.205a" : 13,
    "p.123b" : 14,
    "p.103b" : 15,
    "w.207c" : 16,
    "w.219" : 17,
    "w.209" : 18,
    "w.224" : 19,
    "p.106b" : 20,
    "w.202b" : 21,
    "p.104" : 22,
    "w.205b" : 23,
    "w.210" : 24,
    "w.207b" : 25,
    "dp.135" : 26,
    "p.103a" : 27,
    "r.409" : 28,
    "w.207a" : 29,
    "p.123a" : 30,
    "w.201b" : 31,
    "w.201a" : 32,
    "w.227" : 33,
    "p.124c" : 34,
    "p.131a" : 35,
    "r.301c" : 36,
    "p.115" : 37,
    "w.235" : 38,
    "p.117" : 39,
    "p.124b" : 40,
    "p.245a" : 41,
    "w.203b" : 42,
    "r.302b" : 43,
    "r.407a" : 44,
    "p.107a" : 45,
    "w.205d" : 46,
    "w.233" : 47,
    "r.302a" : 48,
    "r.301e" : 49,
    "p.102" : 50,
    "r.303" : 51,
    "p.124a" : 52,
    "p.106a" : 53,
    "w.225" : 54,
    "p.112" : 55,
    "p.128" : 56
}

Traffic_idx2name = dict(zip([key for key in Traffic_name2idx.values()], [value for value in Traffic_name2idx.keys()]))

def read_ann(ann_path):
    tree = ET.parse(ann_path)
    root = tree.getroot()

    coors = ['xmin', 'ymin', 'xmax', 'ymax']

    bboxes        = []
    labels        = []
    difficulties  = []

    for obj in root.iter('object'):
        # Tên của obj trong box
        name = obj.find('name').text.lower().strip()
        if name not in Traffic_name2idx.keys():
            continue
        labels.append(name)

        # Độ khó 
        difficult = int(obj.find('difficult').text)
        difficulties.append(difficult)

        # Toạ độ
        bnd = obj.find("bndbox")
        box = []
        for coor in coors:
            box.append(int(bnd.find(coor).text))
        bboxes.append(box)

    return bboxes, labels, difficulties

class TrafficSignSVM_dataset(data.Dataset):

    def __init__(self, root_path, split_folder_path, transform=None, phase='train', flag = False):
        super().__init__()
        
        path = os.path.join(root_path, split_folder_path)

        img_pattern = '*.jpg'
        ann_pattern = '*.xml'

        self.img_path_list = []
        self.ann_path_list = []

        ann_temp = glob.glob(path + '/' + ann_pattern)

        for ann_file in ann_temp:
            _1, labels, _2 = read_ann(ann_file)
            if (len(labels) == 0) and (flag == True):
                continue

            self.ann_path_list.append(ann_file)
            img_file = ann_file.replace(".xml", ".jpg")
            self.img_path_list.append(img_file)

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        image                        = cv2.imread(self.img_path_list[index])
        bboxes, labels, difficulties = read_ann(self.ann_path_list[index])
        temp = []
        for label in labels:
            temp.append(Traffic_name2idx[label] - 1)

        bboxes       = np.array(bboxes)
        labels       = np.array(temp)
        difficulties = np.array(difficulties)

        return image, bboxes, labels

def sobel_filters(img):
    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = cv2.filter2D(img, -1, Sx)
    Iy = cv2.filter2D(img, -1, Sy)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return Ix, Iy, G, theta

def calFeatureVector(img, limit_dim = True):
    #img_ = cv2.resize(img, (256, 256)) 
    Ix, Iy, G, theta = sobel_filters(img)

    if not limit_dim:
        return np.hstack((Ix, Iy, G, theta)).flatten()
    else:
        return np.hstack((G, theta)).flatten()

def create_train_set():
    root_path = r'/home/manh/Datasets/Vietnam-Traffic-Sign-Detection.v6i.voc'
    train_path = 'train'
    dataset = TrafficSignSVM_dataset(root_path, train_path)

    train_features = []
    train_labels   = []

    for idx in range(dataset.__len__()):
        img, bboxes, labels = dataset.__getitem__(idx)

        for box, label in zip(bboxes, labels):
            sub_img = img[box[1]:box[3], box[0]:box[2], :].copy()
            sub_img = cv2.resize(sub_img, (50, 50))
            
            feature = calFeatureVector(sub_img)
            train_features.append(feature)
            train_labels.append(label)

    train_features = np.array(train_features)
    train_labels   = np.array(train_labels)

    return train_features, train_labels
    

def featuring(img, bboxes):
    H, W, C = img.shape

    features = []

    if bboxes is not None:
        for box in bboxes:
            box = box.clone().detach()
            xmin = max(0, int(box[0]*W))
            ymin = max(0, int(box[1]*H))
            xmax = min(W, int(box[2]*W))
            ymax = min(H, int(box[3]*H))

            sub_img = img[ymin : ymax, xmin : xmax, :].copy()
            sub_img = cv2.resize(sub_img, (50, 50))

            feature = calFeatureVector(sub_img)
            features.append(feature)

    features = np.array(features)
    return features

def eval(classifier):
    root_path = r'/home/manh/Datasets/Vietnam-Traffic-Sign-Detection.v6i.voc'
    test_path = 'test'
    dataset = TrafficSignSVM_dataset(root_path, test_path)

    test_features = []
    test_labels   = []

    for idx in range(dataset.__len__()):
        img, bboxes, labels = dataset.__getitem__(idx)

        for box, label in zip(bboxes, labels):
            sub_img = img[box[1]:box[3], box[0]:box[2], :].copy()
            sub_img = cv2.resize(sub_img, (50, 50))
            
            feature = calFeatureVector(sub_img)
            test_features.append(feature)
            test_labels.append(label)

    test_features = np.array(test_features)
    test_labels   = np.array(test_labels)

    print(test_features.shape)
    print(test_labels.shape)

    t = set()
    for x in test_labels:
        t.add(x)
    print(t)

    y_pred = classifier.predict(test_features)
    print(f1_score(test_labels, y_pred, average=None))

def create_ML_model():
    train_features, train_labels = create_train_set()
    print(train_features.shape)
    print(train_labels.shape)

    print()
    print("processing...")
    print()

    classifier = svm.SVC()
    classifier.fit(train_features, train_labels)

    print("train done")

    with open('model.pkl', 'wb') as f:
        pickle.dump(classifier, f)

    print("save done")

def ML_model(pretrain_path = '/home/manh/Projects/CS231/weights/model.pkl'):
    if pretrain_path == None:
        return create_ML_model()

    classifier = svm.SVC()
    with open(pretrain_path, 'rb') as f:
        classifier = pickle.load(f)

    return classifier

from utils.box_utils import draw_bounding_box, Non_Maximum_Suppression
from model.SSD300 import SSD300
def detect(num_classes=2, mapping=Traffic_idx2name):

    root_path = r'/home/manh/Datasets/Vietnam-Traffic-Sign-Detection.v6i.voc'
    test_path = 'test'
    dataset = TrafficSignSVM_dataset(root_path, test_path)

    model = SSD300(pretrain_path='/home/manh/Projects/CS231/weights/ML_model_8.pth', n_classes=2)
    model.to("cuda")
    dboxes = model.create_prior_boxes().to("cuda")
    classifier = ML_model()

    for idx in range(dataset.__len__()):
        image, bboxes, labels = dataset.__getitem__(idx)
        origin_image = image.copy()
        
        image        = cv2.resize(image, (300, 300))
        image        = torch.FloatTensor(image[:, :, (2, 1, 0)]).permute(2, 0, 1).contiguous()
        image = image.unsqueeze(0).to("cuda")
        offset, conf = model(image)
        offset = offset.to("cuda")
        conf   = conf.to("cuda")
        pred_bboxes, pred_labels, pred_confs = Non_Maximum_Suppression(dboxes, offset[0], conf[0], conf_threshold=0.3, iou_threshold=0.45, top_k=200, num_classes=num_classes)
        feature = featuring(origin_image, pred_bboxes)
        if feature.shape[0] == 0:
            continue

        y_pred = classifier.predict(feature)
        y_pred = np.array(y_pred)
        y_pred = torch.tensor(y_pred) + 1
  
        draw_bounding_box(origin_image, pred_bboxes, y_pred, pred_labels, mapping)
        cv2.imshow("img", origin_image)
        k = cv2.waitKey()
        if (k == ord('q')):
            break
        #cv2.imwrite(r"H:\test_img\_" + str(idx) + r".jpg", origin_image)
        print("ok")


if __name__ == "__main__":
    
    #classifier = ML_model(pretrain_path='weights/model.pkl')
    #eval(classifier)
    detect()




