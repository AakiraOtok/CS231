from utils.Traffic_utils import TrafficUtils
from utils.augmentations_utils import CustomAugmentation
import cv2

root_path = '/home/manh/Datasets/Vietnam-Traffic-Sign-Detection.v6i.voc'
train_path = 'train'
data = TrafficUtils()
dataset = data.make_dataset(root_path=root_path, split_folder_path=train_path, transform=CustomAugmentation(), phase='train')    
print(dataset.__len__())
origin_image, image, bboxes, labels, difficulties = dataset.__getitem__(0, True)
print(bboxes)
    
cv2.rectangle(origin_image, (int(bboxes[0][0]), int(bboxes[0][1])), (int(bboxes[0][2]), int(bboxes[0][3])), 1, 1, 1)
cv2.imshow('img', origin_image)
cv2.waitKey()