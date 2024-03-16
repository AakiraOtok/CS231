from utils.lib import *
from model.SSD300 import SSD300
from model.SSD512 import SSD512
from model.FPN_SSD300_b import FPN_SSD300
from model.FPN_SSD512 import FPN_SSD512
from utils.VOC_utils import VOCUtils, VOC_idx2name, VOC_name2idx
#from utils.COCO_utils import COCOUtils, COCO_idx2name, COCO_name2idx
from utils.SOHAS_utils import SOHAS_dataset, SOHAS_idx2name
from utils.VEDAI_utils import VEDAI_Utils, VEDAI_idx2name
from utils.box_utils import Non_Maximum_Suppression, draw_bounding_box
from utils.augmentations_utils import CustomAugmentation

def detect(dataset, model, num_classes=21, mapping=VOC_idx2name):
    model.to("cuda")
    dboxes = model.create_prior_boxes().to("cuda")
    #for images, bboxes, labels, difficulties in dataloader:
    for idx in range(dataset.__len__()):
        origin_image, images, bboxes, labels, difficulties = dataset.__getitem__(idx, get_origin_image=True)

        images = images.unsqueeze(0).to("cuda")
        offset, conf = model(images)
        offset = offset.to("cuda")
        conf   = conf.to("cuda")
        pred_bboxes, pred_labels, pred_confs = Non_Maximum_Suppression(dboxes, offset[0], conf[0], conf_threshold=0.3, iou_threshold=0.45, top_k=200, num_classes=num_classes)

        draw_bounding_box(origin_image, pred_bboxes, pred_labels, pred_confs, mapping)
        #cv2.imshow("img", origin_image)
        #cv2.waitKey()
        cv2.imwrite(r"H:\test_img\_" + str(idx) + r".jpg", origin_image)
        print("ok")

def detect_on_COCO(pretrain_path, version = "original", size=300):
    val35k_folder_path = r"H:\data\COCO\val2014"
    val35k_file        = r"H:\data\COCO\instances_valminusminival2014.json"
    dataset = COCOUtils(val35k_folder_path, val35k_file).make_dataset(phase="valid", transform=CustomAugmentation(size=size))

    if version == "original":
        if size==300:
            model      = SSD300(pretrain_path=pretrain_path, data_train_on="COCO",n_classes=81)
        elif size==512:
            model      = SSD512(pretrain_path=pretrain_path, data_train_on="COCO",n_classes=81)
    elif version == "FPN":
        if size == 300:
            model      = FPN_SSD300(pretrain_path=pretrain_path, data_train_on="COCO", n_classes=81)
        elif size == 512:
            model      = FPN_SSD512(pretrain_path=pretrain_path, data_train_on="COCO", n_classes=81)
    
    num_classes = 81
    mapping     = COCO_idx2name

    return dataset, model, num_classes, mapping

def detect_on_VOC(pretrain_path, version="original", size=300):
    data_folder_path = r"H:\projectWPD\data"
    dataset    = VOCUtils(data_folder_path).make_dataset(r"VOC2007", r"test.txt", phase="valid", transform=CustomAugmentation(size=size))

    if version == "origin":
        if size == 300:
            model      = SSD300(pretrain_path=pretrain_path, data_train_on="VOC",n_classes=21)
        elif size == 512:
            model      = SSD512(pretrain_path=pretrain_path, data_train_on="VOC",n_classes=21)
    elif version == "FPN":
        if size == 300:
            model      = FPN_SSD300(pretrain_path=pretrain_path, data_train_on="VOC", n_classes=21)
        elif size == 512:
            model      = FPN_SSD512(pretrain_path=pretrain_path, data_train_on="VOC", n_classes=21)


    num_classes = 21
    mapping     = VOC_idx2name

    return dataset, model, num_classes, mapping

def detect_on_SOHAS(pretrain_path, version="original", size=300):
    data_folder_path = r"H:\data"
    dataset = SOHAS_dataset(data_folder_path, r'test', CustomAugmentation(size=size), phase='valid')

    if version == "origin":
        if size == 300:
            model      = SSD300(pretrain_path=pretrain_path, data_train_on="VOC",n_classes=7)
        elif size == 512:
            model      = SSD512(pretrain_path=pretrain_path, data_train_on="VOC",n_classes=7)
    elif version == "FPN":
        if size == 300:
            model      = FPN_SSD300(pretrain_path=pretrain_path, data_train_on="VOC", n_classes=7)
        elif size == 512:
            model      = FPN_SSD512(pretrain_path=pretrain_path, data_train_on="VOC", n_classes=7)


    num_classes = 7
    mapping     = SOHAS_idx2name

    return dataset, model, num_classes, mapping

def detect_on_VEDAI(pretrain_path, version="original", size=300):
    data_folder_path = r"E:\data"
    dataset = VEDAI_Utils(data_folder_path).make_dataset(r"fold02test.txt", transform=CustomAugmentation(phase='valid'), phase='valid')

    if version == "origin":
        if size == 300:
            model      = SSD300(pretrain_path=pretrain_path, data_train_on="COCO",n_classes=12)
        elif size == 512:
            model      = SSD512(pretrain_path=pretrain_path, data_train_on="COCO",n_classes=12)
    elif version == "FPN":
        if size == 300:
            model      = FPN_SSD300(pretrain_path=pretrain_path, data_train_on="COCO", n_classes=12)
        elif size == 512:
            model      = FPN_SSD512(pretrain_path=pretrain_path, data_train_on="COCO", n_classes=12)


    num_classes = 12
    mapping     = VEDAI_idx2name

    return dataset, model, num_classes, mapping


if __name__ == "__main__":
    #pretrain_path = r"H:\checkpoint\iteration_400000_b_46.27.pth"
    #pretrain_path = r"H:\checkpoint\iteration_400000_c_42.26.pth"
    pretrain_path = r"H:\checkpoint\iteration_118000.pth"
    
    dataset, model, num_classes, mapping = detect_on_VOC(pretrain_path, version="FPN", size=300)
    #dataset, model, num_classes, mapping = detect_on_COCO(pretrain_path, version="FPN", size=300)
    #dataset, model, num_classes, mapping = detect_on_VEDAI(pretrain_path, version="FPN", size=300)
    model.eval()
    
    detect(dataset, model, num_classes=num_classes, mapping=mapping)