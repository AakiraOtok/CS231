from utils.lib import *
from utils.augmentations_utils import CustomAugmentation 

from math import sqrt

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

def collate_fn(batches):
    """
    custom dataset với hàm collate_fn để hợp nhất dữ liệu từ batch_size dữ liệu đơn lẻ

    :param batches được trả về từ __getitem__() 
    
    return:
    images, tensor [batch_size, C, H, W]
    bboxes, list [tensor([n_box, 4]), ...]
    labels, list [tensor([n_box]), ...]
    difficulties, list[tensor[n_box], ...]
    """
    images       = []
    bboxes       = []
    labels       = []
    difficulties = []

    for batch in batches:
        images.append(batch[0])
        bboxes.append(batch[1])
        labels.append(batch[2])
        difficulties.append(batch[3])
    
    images = torch.stack(images, dim=0)
    return images, bboxes, labels, difficulties

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
            box.append(float(bnd.find(coor).text))
        bboxes.append(box)

    return bboxes, labels, difficulties

class TrafficSign_dataset(data.Dataset):

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

        self.transform     = transform
        self.phase         = phase

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index, get_origin_image = False):
        image                        = cv2.imread(self.img_path_list[index])
        origin_image                 = image.copy()
        bboxes, labels, difficulties = read_ann(self.ann_path_list[index])
        temp = []
        for label in labels:
            #temp.append(Traffic_name2idx[label])
            temp.append(1)
        bboxes       = np.array(bboxes)
        labels       = np.array(temp)
        difficulties = np.array(difficulties)

        image, bboxes, labels, difficulties = self.transform(image, bboxes, labels, difficulties, self.phase)

        image        = torch.FloatTensor(image[:, :, (2, 1, 0)]).permute(2, 0, 1).contiguous()
        bboxes       = torch.FloatTensor(bboxes)
        labels       = torch.LongTensor(labels)
        difficulties = torch.LongTensor(difficulties)

        if not get_origin_image:   
            return image, bboxes, labels, difficulties
        else:
            return origin_image, image, bboxes, labels, difficulties
        
class TrafficUtils():

    def make_dataset(self, root_path, split_folder_path, transform, phase, flag=False):
        dataset = TrafficSign_dataset(root_path=root_path, split_folder_path=split_folder_path, transform=transform, phase=phase, flag=flag)
        return dataset

    def make_dataloader(self, root_path, split_folder_path, batch_size, shuffle, transform=CustomAugmentation(), collate_fn=collate_fn, phase='train', num_worker=0, pin_memory=False,  flag=False):
        dataset = self.make_dataset(root_path=root_path, split_folder_path=split_folder_path, transform=transform, phase=phase, flag = flag)
        dataloader = data.DataLoader(dataset, batch_size, shuffle, num_workers=num_worker, collate_fn=collate_fn, pin_memory=pin_memory)
        return dataloader


if __name__ == "__main__":
    root_path = '/home/manh/Datasets/Vietnam-Traffic-Sign-Detection.v6i.voc'
    train_path = 'train'
    dataset = TrafficSign_dataset(root_path=root_path, split_folder_path=train_path)
    print(dataset.__len__())
    origin_image, image, bboxes, labels, difficulties = dataset.__getitem__(0, True)
    print(bboxes)
    
    cv2.rectangle(origin_image, (int(bboxes[0][0]), int(bboxes[0][1])), (int(bboxes[0][2]), int(bboxes[0][3])), 1, 1, 1)
    cv2.imshow('img', origin_image)
    cv2.waitKey()