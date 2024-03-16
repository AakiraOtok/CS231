from utils.lib import *

class VGG16Base(nn.Module):
    """
    Lấy VGG16 làm base network, tuy nhiên cần có một vài thay đổi:
    - Đầu vào ảnh là 300x300 thay vì 224x224, các comment bên dưới sẽ áp dụng cho đầu vào 300x300
    - Lớp pooling thứ 3 sử dụng ceiling mode thay vì floor mode
    - Lớp pooling thứ 5 kernel size (2, 2) -> (3, 3) và stride 2 -> 1, và padding = 1
    - Ta downsample (decimate) parameter fc6 và fc7 để tạo thành conv6 và conv7, loại bỏ hoàn toàn fc8
    """

    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(in_channels=  3, out_channels= 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size=3, padding=1)
        self.pool1   = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels= 64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2   = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3   = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4   = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5   = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # Không còn fc layers nữa, thay vào đó là conv6 và conv7
        # atrous
        self.conv6   = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
        self.conv7   = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)

    def decimate(self, tensor, steps):
        assert(len(steps) == tensor.dim())
        
        for i in range(tensor.dim()):
            if steps[i] is not None:
                tensor = tensor.index_select(dim=i, index=torch.arange(start=0, end=tensor.shape[i], step=steps[i]))

        return tensor

    
    def load_pretrain(self):
        """
        load pretrain từ thư viện pytorch, decimate param lại để phù hợp với conv6 và conv7
        """

        state_dict  = self.state_dict() 
        param_names = list(state_dict.keys())

        # old version : torch.vision.models.vgg16(pretrain=True)
        # Load model theo API mới của pytorch, cụ thể hơn tại : https://pytorch.org/vision/stable/models.html
        pretrain_state_dict  = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT').state_dict()
        pretrain_param_names = list(pretrain_state_dict.keys())

        # Pretrain param name và custom param name không giống nhau, các param chỉ cùng thứ tự như trong architecture
        for idx, param_name in enumerate(param_names[:-4]): # 4 param cuối là weight và bias của conv6 và conv7, sẽ xử lí sau
            state_dict[param_name] = pretrain_state_dict[pretrain_param_names[idx]]

        # fc -> conv
        fc6_weight = pretrain_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        fc6_bias   = pretrain_state_dict['classifier.0.bias'].view(4096)

        fc7_weight = pretrain_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        fc7_bias   = pretrain_state_dict['classifier.3.bias'].view(4096)

        # downsample parameter
        state_dict['conv6.weight'] = self.decimate(fc6_weight, steps=[4, None, 3, 3])
        state_dict['conv6.bias']   = self.decimate(fc6_bias, steps=[4])

        state_dict['conv7.weight'] = self.decimate(fc7_weight, steps=[4, 4, None, None])
        state_dict['conv7.bias']   = self.decimate(fc7_bias, steps=[4])

        self.load_state_dict(state_dict)


    def forward(self, images):
        """
        :param images, tensor [N, 3, 300, 300]

        return:
        """
        out = F.relu(self.conv1_1(images)) # [N, 64, 300, 300]
        out = F.relu(self.conv1_2(out))    # [N, 64, 300, 300]
        out = self.pool1(out)              # [N, 64, 150, 150]

        out = F.relu(self.conv2_1(out))    # [N, 128, 150, 150]
        out = F.relu(self.conv2_2(out))    # [N, 128, 150, 150]
        out = self.pool2(out)              # [N, 128, 75, 75]

        out = F.relu(self.conv3_1(out))    # [N, 256, 75, 75]
        out = F.relu(self.conv3_2(out))    # [N, 256, 75, 75]
        out = F.relu(self.conv3_3(out))    # [N, 256, 75, 75]
        out = self.pool3(out)              # [N, 256, 38, 38] không phải [N, 256, 37, 37] do ceiling mode = True

        out = F.relu(self.conv4_1(out))    # [N, 512, 38, 38]
        out = F.relu(self.conv4_2(out))    # [N, 512, 38, 38]
        out = F.relu(self.conv4_3(out))    # [N, 512, 38, 38]
        conv4_3_feats = out                # [N, 512, 38, 38]
        out = self.pool4(out)              # [N, 512, 19, 19]

        out = F.relu(self.conv5_1(out))    # [N, 512, 19, 19]
        out = F.relu(self.conv5_2(out))    # [N, 512, 19, 19]
        out = F.relu(self.conv5_3(out))    # [N, 512, 19, 19]
        out = self.pool5(out)              # [N, 512, 19, 19], layer pooling này không làm thay đổi size features map

        out = F.relu(self.conv6(out))      # [N, 1024, 19, 19]

        conv7_feats = F.relu(self.conv7(out)) # [N, 1024, 19, 19]

        return conv4_3_feats, conv7_feats
    

class AuxiliraryConvolutions(nn.Module):
    """ Sau base network (vgg16) sẽ là các lớp conv phụ trợ
    Feature Pyramid Network
    """

    def __init__(self):
        super().__init__()
        
        self.conv8_1  = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, padding=0)
        self.conv8_2  = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        
        self.conv9_1  = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=0)
        self.conv9_2  = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0)

        self.conv11_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0)

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                if c.bias is not None:
                    nn.init.constant_(c.bias, 0.)


    def forward(self, conv7_feats):
        """
        :param conv8_feats, tensor [N, 1024, 19, 19]
        """

        out = F.relu(self.conv8_1(conv7_feats))   # [N, 256, 19, 19]
        out = F.relu(self.conv8_2(out))           # [N, 512, 10, 10]
        conv8_2_feats = out                       # [N, 512, 10, 10]

        out = F.relu(self.conv9_1(out))           # [N, 128, 10, 10]
        out = F.relu(self.conv9_2(out))           # [N, 256, 5, 5]
        conv9_2_feats = out                       # [N, 256, 5, 5]

        out = F.relu(self.conv10_1(out))          # [N, 128, 5, 5]
        out = F.relu(self.conv10_2(out))          # [N, 256, 3, 3]
        conv10_2_feats = out                      # [N, 256, 3, 3]

        out = F.relu(self.conv11_1(out))          # [N, 128, 3, 3]
        conv11_2_feats = F.relu(self.conv11_2(out))          # [N, 256, 1, 1]

        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats
    
class FPNConvolutions(nn.Module):
    """ 
    conv3_3_feats  : [N, 256, 75, 75]
    conv4_3_feats  : [N, 512, 38, 38]
    conv7_feats    : [N, 1024, 19, 19]
    conv8_2_feats  : [N, 512, 10, 10]
    conv9_2_feats  : [N, 256, 5, 5]
    conv10_2_feats : [N, 256, 3, 3]
    conv11_2_feats : [N, 256, 1, 1]
    """

    def __init__(self):
        super().__init__()

        self.fp5_upsample = nn.Upsample(scale_factor=3, mode="bilinear")
        self.fp5_conv1    = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, bias=False)
        self.fp5_bn       = nn.BatchNorm2d(num_features=256)

        self.fp4_upsample = nn.Upsample(scale_factor=5/3, mode="bilinear")
        self.fp4_conv1    = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, bias=False)
        self.fp4_bn       = nn.BatchNorm2d(num_features=256)

        self.fp3_upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.fp3_conv1    = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, bias=False)
        self.fp3_bn       = nn.BatchNorm2d(num_features=512)

        self.fp2_upsample = nn.Upsample(scale_factor=1.9, mode="bilinear")
        self.fp2_conv1    = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, bias=False)
        self.fp2_bn       = nn.BatchNorm2d(num_features=1024)

        self.fp1_upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.fp1_conv1    = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bias=False)
        self.fp1_bn       = nn.BatchNorm2d(num_features=512)


    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                if c.bias is not None:
                    nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats ,conv11_2_feats):

        fp6_feats = conv11_2_feats

        out = self.fp5_upsample(conv11_2_feats)
        out = F.relu(F.relu(self.fp5_conv1(out)) + conv10_2_feats)
        fp5_feats = self.fp5_bn(out)

        out = self.fp4_upsample(out)
        out = F.relu(F.relu(self.fp4_conv1(out)) + conv9_2_feats)
        fp4_feats = self.fp4_bn(out)

        out = self.fp3_upsample(out)
        out = F.relu(F.relu(self.fp3_conv1(out)) + conv8_2_feats)
        fp3_feats = self.fp3_bn(out)

        out = self.fp2_upsample(out)
        out = F.relu(F.relu(self.fp2_conv1(out)) + conv7_feats)
        fp2_feats = self.fp2_bn(out)

        out = self.fp1_upsample(out)
        out = F.relu(F.relu(self.fp1_conv1(out)) + conv4_3_feats)
        fp1_feats = self.fp1_bn(out)

        return fp1_feats, fp2_feats, fp3_feats, fp4_feats, fp5_feats, fp6_feats

class PredictionConvolutions(nn.Module):
    """Layer cuối là để predict offset và conf

    """

    def __init__(self, n_classes=21):
        super().__init__()

        self.n_classes = n_classes

        n_boxes={
            'fp1' : 4,
            'fp2' : 6,
            'fp3' : 6,
            'fp4' : 6,
            'fp5' : 4,
            'fp6' : 4
        }

        # kernel size = 3 và padding = 1 không làm thay đổi kích thước feature map 

        self.loc_fp6  = nn.Conv2d(256,   n_boxes['fp6']*4, kernel_size=3, padding=1)
        self.loc_fp5  = nn.Conv2d(256,   n_boxes['fp5']*4, kernel_size=3, padding=1)
        self.loc_fp4  = nn.Conv2d(256,   n_boxes['fp4']*4, kernel_size=3, padding=1)
        self.loc_fp3  = nn.Conv2d(512,   n_boxes['fp3']*4, kernel_size=3, padding=1)
        self.loc_fp2  = nn.Conv2d(1024,   n_boxes['fp2']*4, kernel_size=3, padding=1)
        self.loc_fp1  = nn.Conv2d(512,   n_boxes['fp1']*4, kernel_size=3, padding=1)


        self.conf_fp6  = nn.Conv2d(256,  n_boxes['fp6']*n_classes, kernel_size=3, padding=1)
        self.conf_fp5  = nn.Conv2d(256,  n_boxes['fp5']*n_classes, kernel_size=3, padding=1)
        self.conf_fp4  = nn.Conv2d(256,  n_boxes['fp4']*n_classes, kernel_size=3, padding=1)
        self.conf_fp3  = nn.Conv2d(512,  n_boxes['fp3']*n_classes, kernel_size=3, padding=1)
        self.conf_fp2  = nn.Conv2d(1024,  n_boxes['fp2']*n_classes, kernel_size=3, padding=1)
        self.conf_fp1  = nn.Conv2d(512,  n_boxes['fp1']*n_classes, kernel_size=3, padding=1)

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                if c.bias is not None:
                    nn.init.constant_(c.bias, 0.)

    def forward(self, fp1_feats, fp2_feats, fp3_feats, fp4_feats, fp5_feats, fp6_feats):

        batch_size = fp1_feats.shape[0]


        loc_fp1   = self.loc_fp1(fp1_feats)
        loc_fp1   = loc_fp1.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        
        loc_fp2     = self.loc_fp2(fp2_feats)
        loc_fp2     = loc_fp2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        loc_fp3   = self.loc_fp3(fp3_feats)
        loc_fp3   = loc_fp3.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        loc_fp4   = self.loc_fp4(fp4_feats)
        loc_fp4   = loc_fp4.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        loc_fp5   = self.loc_fp5(fp5_feats)
        loc_fp5   = loc_fp5.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        loc_fp6   = self.loc_fp6(fp6_feats)
        loc_fp6   = loc_fp6.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)



        conf_fp1   = self.conf_fp1(fp1_feats)
        conf_fp1   = conf_fp1.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes)
        
        conf_fp2     = self.conf_fp2(fp2_feats)
        conf_fp2     = conf_fp2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes)

        conf_fp3   = self.conf_fp3(fp3_feats)
        conf_fp3   = conf_fp3.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes)

        conf_fp4   = self.conf_fp4(fp4_feats)
        conf_fp4   = conf_fp4.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes)

        conf_fp5   = self.conf_fp5(fp5_feats)
        conf_fp5   = conf_fp5.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes)

        conf_fp6   = self.conf_fp6(fp6_feats)
        conf_fp6   = conf_fp6.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes)


        loc  = torch.cat((loc_fp1, loc_fp2, loc_fp3, loc_fp4, loc_fp5, loc_fp6), dim=1)
        conf = torch.cat((conf_fp1, conf_fp2, conf_fp3, conf_fp4, conf_fp5, conf_fp6), dim=1)

        return loc, conf
        
    
class L2Norm(nn.Module):
    def __init__(self, input_channel, scale=20.):
        super().__init__()
        self.scale_factors = nn.Parameter(torch.FloatTensor(1, input_channel, 1, 1))
        self.eps           = 1e-10
        nn.init.constant_(self.scale_factors, scale)
    
    def forward(self, tensor):
        norm   = tensor.pow(2).sum(dim=1, keepdim=True).sqrt()
        tensor = tensor/(norm + self.eps)*self.scale_factors
        return tensor
    
class FPN_SSD300(nn.Module):

    def __init__(self, pretrain_path = None, data_train_on = "VOC", n_classes = 21):
        super().__init__()

        self.n_classes   = n_classes
        self.data_train_on = data_train_on
        self.base_net    = VGG16Base()
        self.auxi_conv   = AuxiliraryConvolutions()
        self.fp_conv     = FPNConvolutions()
        self.pred_conv   = PredictionConvolutions(n_classes) 
        self.l2_conv4_3  = L2Norm(input_channel=512)

        if pretrain_path is not None:
            self.load_state_dict(torch.load(pretrain_path))
        else:
            self.base_net.load_pretrain()
            self.auxi_conv.init_conv2d()
            self.fp_conv.init_conv2d()
            self.pred_conv.init_conv2d()

    def create_prior_boxes(self):
        """ 
        mỗi box có dạng [cx, cy, w, h] được scale
        """
        # kích thước feature map tương ứng
        fmap_sizes    = [38, 19, 10, 5, 3, 1]
        
        # scale như trong paper và được tính sẵn thay vì công thức
        # lưu ý ở conv4_3, tác giả xét như một trường hợp đặc biệt (scale 0.1):
        # Ở mục 3.1, trang 7 : 
        # "We set default box with scale 0.1 on conv4 3 .... "
        # "For SSD512 model, we add extra conv12 2 for prediction, set smin to 0.15, and 0.07 on conv4 3...""

        if self.data_train_on == "VOC":
            box_scales    = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
        elif self.data_train_on == "COCO":
            box_scales    = [0.07, 0.15, 0.3375, 0.525, 0.7125, 0.9] 
            
        aspect_ratios = [
                [1., 2., 0.5],
                [1., 2., 3., 0.5, 0.333],
                [1., 2., 3., 0.5, 0.333],
                [1., 2., 3., 0.5, 0.333],
                [1., 2., 0.5],
                [1., 2., 0.5]
            ]
        dboxes = []
        
        
        for idx, fmap_size in enumerate(fmap_sizes):
            for i in range(fmap_size):
                for j in range(fmap_size):

                    # lưu ý, cx trong ảnh là trục hoành, do đó j + 0.5 chứ không phải i + 0.5
                    cx = (j + 0.5) / fmap_size
                    cy = (i + 0.5) / fmap_size

                    for aspect_ratio in aspect_ratios[idx]:
                        scale = box_scales[idx]
                        dboxes.append([cx, cy, scale*sqrt(aspect_ratio), scale/sqrt(aspect_ratio)])

                        if aspect_ratio == 1.:
                            try:
                                scale = sqrt(scale*box_scales[idx + 1])
                            except IndexError:
                                scale = 1.
                            dboxes.append([cx, cy, scale*sqrt(aspect_ratio), scale/sqrt(aspect_ratio)])

        dboxes = torch.FloatTensor(dboxes)
        
        #dboxes = pascalVOC_style(dboxes)
        dboxes.clamp_(0, 1)
        #dboxes = yolo_style(dboxes)
                
        return dboxes

    def forward(self, images):
        conv4_3_feats, conv7_feats                         = self.base_net(images)
        conv4_3_feats                                                     = self.l2_conv4_3(conv4_3_feats)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats      = self.auxi_conv(conv7_feats)

        FP1_feats, FP2_feats, FP3_feats, FP4_feats, FP5_feats, FP6_feats = self.fp_conv(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)

        loc, conf  = self.pred_conv(FP1_feats, FP2_feats, FP3_feats, FP4_feats, FP5_feats, FP6_feats)
        return loc, conf 



if __name__ == "__main__":
    T = FPN_SSD300()
    img = torch.ones(1, 3, 300, 300)
    loc, conf = T(img)
    print(loc.shape)
    print(conf.shape)