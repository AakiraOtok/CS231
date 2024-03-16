from utils.lib import *
from model.SSD300 import SSD300
from model.SSD512 import SSD512
from model.FPN_SSD300_b import FPN_SSD300
from model.FPN_SSD512 import FPN_SSD512
from utils.Traffic_utils import TrafficUtils, Traffic_idx2name
from utils.box_utils import Non_Maximum_Suppression, draw_bounding_box
from utils.augmentations_utils import CustomAugmentation

def detect(dataset, model, num_classes=21, mapping=Traffic_idx2name):
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
        cv2.imshow("img", origin_image)
        k = cv2.waitKey()
        if (k == ord('q')):
            break
        #cv2.imwrite(r"H:\test_img\_" + str(idx) + r".jpg", origin_image)
        print("ok")

def detect_on_Traffic(size=300, version="original", pretrain_path=None):
    root_path = r'H:\Datasets'
    test_path = 'test'

    data = TrafficUtils()
    dataset       = data.make_dataset(root_path=root_path, split_folder_path=test_path, transform=CustomAugmentation(phase='valid'), phase='valid', flag = True)

    if version == "original":
        model = SSD300(n_classes=57, pretrain_path=pretrain_path)
    elif version == "FPN":
        model = FPN_SSD300(n_classes=57, pretrain_path=pretrain_path)
        
    num_classes = 57
    mapping = Traffic_idx2name
    return dataset, model, num_classes, mapping

if __name__ == "__main__":
    pretrain_path = r"H:\checkpoint\iteration_43000.pth"
    
    dataset, model, num_classes, mapping = detect_on_Traffic(pretrain_path=pretrain_path, version="FPN", size=300)
    model.eval()
    
    detect(dataset, model, num_classes=num_classes, mapping=mapping)