from utils.lib import *
from utils.augmentations_utils import CustomAugmentation

VOC_name2idx = {
    'aeroplane'   : 1,
    'bicycle'     : 2,
    'bird'        : 3,
    'boat'        : 4,
    'bottle'      : 5, 
    'bus'         : 6,
    'car'         : 7,
    'cat'         : 8,
    'chair'       : 9,
    'cow'         : 10,
    'diningtable' : 11,
    'dog'         : 12,
    'horse'       : 13,
    'motorbike'   : 14,
    'person'      : 15,
    'pottedplant' : 16,
    'sheep'       : 17,
    'sofa'        : 18,
    'train'       : 19,
    'tvmonitor'   : 20
}

# map ngược lại
VOC_idx2name = dict(zip([key for key in VOC_name2idx.values()], [value for value in VOC_name2idx.keys()]))


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
    """
    Đọc thông tin trong annotation

    args:
    ann_path : path của file xml cần đọc

    return:
    bboxes     : list [[xmin, ymin, xmax, ymax], ...]
    labels     : list ['dog', 'cat', ...]
    difficults : list [0, 1, 1, 0 ...]
    """

    tree = ET.parse(ann_path)
    root = tree.getroot()

    coors = ['xmin', 'ymin', 'xmax', 'ymax']

    bboxes        = []
    labels        = []
    difficulties  = []

    for obj in root.iter('object'):
        # Tên của obj trong box
        name = obj.find('name').text.lower().strip()
        labels.append(name)

        # Độ khó 
        difficult = int(obj.find('difficult').text)
        difficulties.append(difficult)

        # Toạ độ
        bnd = obj.find("bndbox")
        box = []
        for coor in coors:
            box.append(float(bnd.find(coor).text) - 1)
        bboxes.append(box)

    return bboxes, labels, difficulties

class VOC_dataset(data.Dataset):
    """_summary_

    Args:
        data (_type_): _description_
    """

    def __init__(self, img_path_list, ann_path_list, transform, phase):
        super().__init__()
        self.img_path_list = img_path_list
        self.ann_path_list = ann_path_list
        self.transform     = transform
        self.phase         = phase
    
    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index, get_origin_image=False):
        image                        = cv2.imread(self.img_path_list[index])
        origin_image                 = image.copy()
        bboxes, labels, difficulties = read_ann(self.ann_path_list[index])
        temp = []
        for label in labels:
            temp.append(VOC_name2idx[label])
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


class VOCUtils():
    """
    Thực hiện việc tạo dataset và dataloader cho VOC dataset
    """

    def __init__(self, data_folder_path):
        """
        data_folder_path : nơi tải VOCdevkit
        """
        self.data_folder_path = data_folder_path


    def make_data_path_list(self, version, file_txt):
        """
        Tạo list image path và annotation path

        :param version  : phiên bản của VOC (VOC2007, VOC2012 .etc)
        :param file_txt : file id của ảnh (trainval.txt, test.txt .etc)

        return:
        Hai list image_path, ann_path
        """
        
        img_template = os.path.join(self.data_folder_path, r"VOCdevkit", version, r"JPEGImages",  r"{}.jpg")
        ann_template = os.path.join(self.data_folder_path, r"VOCdevkit", version, r"Annotations", r"{}.xml")
        
        file_txt_path = os.path.join(self.data_folder_path, r"VOCdevkit", version, r"ImageSets", r"Main", file_txt)

        img_list = []
        ann_list = []

        for line in open(file_txt_path):
            id = line.strip()
            img_list.append(img_template.format(id))
            ann_list.append(ann_template.format(id))
            

        return img_list, ann_list

    def make_dataset(self, version, file_txt, transform=CustomAugmentation(), phase='train'):
        img_path_list, ann_path_list = self.make_data_path_list(version, file_txt)
        dataset = VOC_dataset(img_path_list, ann_path_list, transform, phase)
        return dataset
    
    def make_dataloader(self, version, file_txt, batch_size, shuffle, transform=CustomAugmentation(), collate_fn=collate_fn, phase='train',num_worker=0, pin_memory=False):
        dataset    = self.make_dataset(version, file_txt, transform, phase)
        dataloader = data.DataLoader(dataset, batch_size, shuffle, num_workers=num_worker, collate_fn=collate_fn, pin_memory=pin_memory) 
        return dataloader

if __name__ == "__main__":
    data_folder_path = r"H:\projectWPD\data"
    voc = VOCUtils(data_folder_path)
    dataloader = voc.make_dataloader(r"VOC2007", r"test.txt", 1, 0, phase="train")
    for images, bboxes, labels, difficulties in dataloader:
        img = images.squeeze(0).permute(1, 2, 0).contiguous()[:, :, (2, 1, 0)].cpu().numpy()
        bboxes = bboxes[0]
        labels = labels[0]
        difficulties = difficulties[0]
        
        H, W, C = img.shape
        H -= 1
        W -= 1

        for box, label, difficult in zip(bboxes, labels, difficulties):
            p1 = (int(box[0]*W), int(box[1]*H))
            p2 = (int(box[2]*W), int(box[3]*H))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 1)
        cv2.imshow("img", img)
        cv2.waitKey()

