from utils.lib import *
from utils.Traffic_utils import TrafficUtils
from model.SSD300 import SSD300
from model.SSD512 import SSD512
from model.FPN_SSD300_b import FPN_SSD300
from model.FPN_SSD512 import FPN_SSD512
from utils.box_utils import MultiBoxLoss
from utils.augmentations_utils import CustomAugmentation

def warmup_learning_rate(optimizer, epoch, lr):
    lr_init = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_init + (lr - lr_init)*epoch/5

def train_model(dataloader, model, criterion, optimizer, adjustlr_schedule=(80000, 100000), max_iter=200000):
    torch.backends.cudnn.benchmark = True
    model.to("cuda")
    dboxes = model.create_prior_boxes().to("cuda")
    iteration = -1

    while(1):
        for batch_images, batch_bboxes, batch_labels, batch_difficulties in dataloader: 
            iteration += 1
            t_batch = time.time()

            if iteration in adjustlr_schedule:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1

            batch_size   = batch_images.shape[0]
            batch_images = batch_images.to("cuda")
            for idx in range(batch_size):
                batch_bboxes[idx]       = batch_bboxes[idx].to("cuda")
                batch_labels[idx]       = batch_labels[idx].to("cuda")
                batch_difficulties[idx] = batch_difficulties[idx].to("cuda")

            loc, conf = model(batch_images)

            loss = criterion(loc, conf, dboxes, batch_bboxes, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
            optimizer.step()

            print("iteration : {}, time = {}, loss = {}".format(iteration + 1, round(time.time() - t_batch, 2), loss))
                # save lại mỗi 1000 iteration
            if (iteration + 1) % 1000 == 0:
                torch.save(model.state_dict(), r"/home/manh/checkpoint/iteration_" + str(iteration + 1) + ".pth")
                print("Saved model at iteration : {}".format(iteration + 1))
                if iteration + 1 == max_iter:
                    sys.exit()

def train_on_Traffic(size=300, version="original", pretrain_path=None):
    root_path = '/home/manh/Datasets/Vietnam-Traffic-Sign-Detection.v6i.voc'
    train_path = 'train'

    data = TrafficUtils()
    dataloader       = data.make_dataloader(root_path=root_path, split_folder_path=train_path, batch_size=32
                                            ,shuffle = True, num_worker=6, pin_memory=True, flag=True)

    if version == "original":
        model = SSD300(n_classes=57, pretrain_path=pretrain_path)
    elif version == "FPN":
        model = FPN_SSD300(n_classes=57, pretrain_path=pretrain_path)

    criterion  = MultiBoxLoss(num_classes=57)

    return dataloader, model, criterion

if __name__ == "__main__":

    #########################################################
    #size = 300

    #data_folder_path = r"H:\projectWPD\data"
    #voc              = VOCUtils(data_folder_path)
    
    #dataset1         = voc.make_dataset(r"VOC2007", r"trainval.txt", transform=CustomAugmentation(size=size))
    #dataset2         = voc.make_dataset(r"VOC2012", r"trainval.txt", transform=CustomAugmentation(size=size))
    #dataset          = data.ConcatDataset([dataset1, dataset2])

    #dataloader       = data.DataLoader(dataset, 32, True, num_workers=6, collate_fn=collate_fn, pin_memory=True)
    #criterion  = MultiBoxLoss(num_classes=21)
    #model = augFPN_SSD300(n_classes=21)
    #########################################################

    #pretrain_path = r"H:\checkpoint\iteration_120000_b_78.29.pth"
    pretrain_path = None
    dataloader, model, criterion = train_on_Traffic(version="original", size=300, pretrain_path=pretrain_path)
    biases     = []
    not_biases = []
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer  = optim.SGD(params=[{'params' : biases, 'lr' : 2 * 1e-4}, {'params' : not_biases}], lr=1e-4, momentum=0.9, weight_decay=5e-4)



    train_model(dataloader, model, criterion, optimizer, adjustlr_schedule=(20000, 35000), max_iter=45000)