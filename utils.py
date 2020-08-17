# original code: https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py

import torch
import random
import numpy as np
from collections import defaultdict

__all__ = ["Compose", "Lighting", "ColorJitter"]


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)

def cutmix_data(input, target, cutmix_beta):
    lam = np.random.beta(cutmix_beta, cutmix_beta)
    rand_index = torch.randperm(input.size()[0]).cuda()
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

    return input, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def check_version(cifar_version):
    if cifar_version not in ['10', '100', '20']:
        raise ValueError('cifar version must be one of 10, 20, 100.')

def img_num(cifar_version):
    check_version(cifar_version)
    dt = {'10': 5000, '100': 500, '20': 2500}
    return dt[cifar_version]

def get_oversample(max_class, img_list):
    print(len(img_list))
    if len(img_list)==max_class:
        return img_list
    indx_list= list(range(0, len(img_list)))
    indx_list= pd.DataFrame({'a':indx_list})
    indx_list= indx_list.sample(max_class, replace=True, random_state=42)
    indx_list= indx_list['a'].values.tolist()
    img_list_return=[]
    for indx in indx_list:
        img_list_return.append(img_list[indx])
    return img_list_return

def get_undersample(min_class, img_list):
    print(len(img_list))
    if len(img_list)<=min_class:
        return img_list
    img_list= img_list[:min_class]
    return img_list

def get_imbalanced_data(args, train_data):
    """Get a list of imbalanced training data, store it into im_data dict."""
    cifar_version= args.dataset[5:]
    img_num_per_cls= get_img_num_per_cls(cifar_version, args.imb_factor)
    im_data = defaultdict(list)
    print(train_data.targets)

    for idx,train_label in enumerate(train_data.targets):
        im_data[train_label].append(train_data.data[idx])

    for cls_idx, img_id_list in im_data.items():
        random.seed(args.seed)
        random.shuffle(img_id_list)
        img_num = img_num_per_cls[int(cls_idx)]
        img_list= img_id_list[:img_num]
        if args.sample=="oversample":
            print("oversampling")
            img_list = get_oversample(max(img_num_per_cls), img_list)
        elif args.sample=="undersample":
            print("undersampling")
            img_list=get_undersample(min(img_num_per_cls), img_list)
        im_data[cls_idx]= img_list
        print(np.array(img_list).shape)

    temp_list=[]
    for cls_idx, img_id_list in im_data.items():
        for img in img_id_list:
            temp_list.append((cls_idx, img))
        # random.shuffle(img_list)
        targets_temp, data_temp= zip(*temp_list)
        targets_temp, data_temp= list(targets_temp), np.array(data_temp)
        train_data.data= data_temp
        train_data.targets= targets_temp
        # os.makedirs('data_verify/{}/'.format(cls_idx), exist_ok=True)
        # for i in range(50):
        #     cv2.imwrite('data_verify/{}/'.format(cls_idx)+'{}_image.png'.format(i), img_list[i])

    return train_data



def get_img_num_per_cls(cifar_version, imb_factor=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """

    print("Creating imbalance for cifar version {}".format(cifar_version))
    cls_num = int(cifar_version)
    img_max = img_num(cifar_version)
    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls
