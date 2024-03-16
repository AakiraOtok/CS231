from utils.lib import *
from model.SSD300 import SSD300
from model.SSD512 import SSD512
from model.FPN_SSD300_b import FPN_SSD300
from utils.box_utils import draw_bounding_box, Non_Maximum_Suppression
from utils.VOC_utils import VOC_name2idx, VOC_idx2name
from utils.live_aug_utils import CustomAugmentation
from collections import deque

def live_cam(model, cam, size=300, num_classes=21, mapping = VOC_idx2name):
    model.to("cuda")
    dboxes = model.create_prior_boxes().to("cuda")
    d = deque()

    while cv2.waitKey(1) != ord('q'):
        ret, img = cam.read()
        if not ret:
            break

        aug = CustomAugmentation(size=size, phase='valid')
    
        transformed_img, _1, _2, _3 = aug(img, None, None, None, phase="valid")
        transformed_img        = torch.FloatTensor(transformed_img[:, :, (2, 1, 0)]).permute(2, 0, 1).contiguous().to("cuda")

        offset, conf = model(transformed_img.unsqueeze(0))
        
        pred_bboxes, pred_labels, pred_confs = Non_Maximum_Suppression(dboxes, offset[0], conf[0], conf_threshold=0.2, iou_threshold=0.45, top_k=200, num_classes=num_classes)
        draw_bounding_box(img, pred_bboxes, pred_labels, pred_confs, mapping)
        H, W, C = img.shape
        
        t = time.time()
        d.append(t)
        if len(d) > 1000:
            d.popleft()

        if len(d) >= 100:
            fps = str(int(len(d)/(d[-1] - d[0])))
        else:
            fps = "Measuring" 
        cv2.putText(img, "FPS : " + fps, (int(W - W*20/100), int(H*5/100)), 1, 1, (0, 255, 0), 1)
        cv2.imshow("img", img)

if __name__ == "__main__":
    pretrain_path = r"C:\Users\eguit\Documents\iteration_120000_SSD300_77.2.pth"
    n_classes     = 21

    #model = FPN_SSD300(pretrain_path, n_classes=n_classes)
    model = SSD300(pretrain_path, n_classes=n_classes)
    model.eval()

    cam   = cv2.VideoCapture(0)
    live_cam(model, cam, size=300)