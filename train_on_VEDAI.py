from utils.lib import *
from model.SSD300 import SSD300
from model.SSD512 import SSD512
from model.FPN_SSD300_b import FPN_SSD300
from model.FPN_SSD512 import FPN_SSD512
from utils.box_utils import MultiBoxLoss
from utils.augmentations_utils import CustomAugmentation
from utils.VEDAI_utils import VEDAI_collate_fn, VEDAI_Utils

def warmup_learning_rate(optimizer, epoch, lr):
    lr_init = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_init + (lr - lr_init)*epoch/5

def train_model(dataloader, model, criterion, optimizer, len_data, file_save, adjustlr_schedule=(80000, 100000), max_iter=200000):
    torch.backends.cudnn.benchmark = True
    model.to("cuda")
    dboxes = model.create_prior_boxes().to("cuda")
    iteration = -1

    while(1):
        for batch_images, batch_bboxes, batch_labels, batch_difficulties in dataloader: 
            iteration += 1
            t_batch = time.time()

            epoch = iteration/len_data

            if epoch <= 5:
                warmup_learning_rate(optimizer=optimizer, lr=1e-3 , epoch=epoch)

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
                # save lại mỗi 10000 iteration
            if (iteration + 1) % 1000 == 0:
                torch.save(model.state_dict(), r"E:\checkpoint\{}".format(file_save) + '_' + str(iteration + 1) + ".pth")
                print("Saved model at iteration : {}".format(iteration + 1))
                if iteration + 1 == max_iter:
                    return

fold_list = ["fold02.txt", "fold03.txt", "fold04.txt", "fold05.txt", "fold06.txt", "fold07.txt", "fold08.txt", "fold09.txt", "fold10.txt"]
fold_test_list = ["fold01test.txt", "fold02test.txt", "fold03test.txt", "fold04test.txt", "fold05test.txt", "fold06test.txt", "fold07test.txt", "fold08test.txt", "fold09test.txt", "fold10test.txt"]

if __name__ == "__main__":

    size = 300
    num_classes = 12
    data_folder_path = r"E:\data"

    for foldfile in fold_list:
        vedai            = VEDAI_Utils(data_folder_path)
        dataset          = vedai.make_dataset(foldfile, transform=CustomAugmentation(size=size))
        len_data         = dataset.__len__()
        dataloader       = data.DataLoader(dataset, 8, True, num_workers=2, collate_fn=VEDAI_collate_fn, pin_memory=True)
        criterion        = MultiBoxLoss(num_classes=num_classes)
        model            = FPN_SSD300(n_classes=num_classes, data_train_on="COCO")
 
        biases     = []
        not_biases = []
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        optimizer  = optim.SGD(params=[{'params' : biases, 'lr' : 4 * 1e-4}, {'params' : not_biases}], lr=2 * 1e-4, momentum=0.9, weight_decay=5e-4)
        train_model(dataloader, model, criterion, optimizer, file_save=foldfile.split('.')[0], adjustlr_schedule=(20000, 35000), max_iter=45000, len_data=len_data)