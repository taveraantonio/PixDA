import os
import numpy as np
from torch.utils import data
from PIL import Image
from dataset.utils import encode_segmap
import json


class CityscapesDataset(data.Dataset):

    def __init__(self, root, mean, crop_size, train=True, max_iters=None, ignore_index=255, num_shot=1):
        self.root = root
        self.mean = mean
        self.crop_size = crop_size
        self.train = train
        self.set = 'train' if self.train else 'val'
        self.ignore_index = ignore_index
        self.num_shot = num_shot
        self.files = []
        if self.train:
            self.img_ids = [i_id.strip() for i_id in open('./dataset/cityscapes_list/train_%sshot.txt' % self.num_shot)]
        else:
            self.img_ids = [i_id.strip() for i_id in open('./dataset/cityscapes_list/val.txt')]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.info = json.load(open('./dataset/cityscapes_list/info.json', 'r'))
        self.class_mapping = self.info['label2train']

        for name in self.img_ids:
            image_path = os.path.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            label_path = os.path.join(self.root, "gtFine/%s/%s"
                                      % (self.set, name.replace('_leftImg8bit.png', '_gtFine_labelIds.png')))
            self.files.append({
                "image": image_path,
                "label": label_path,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]

        # open image and label file
        image = Image.open(file['image']).convert('RGB')
        label = Image.open(file['label'])
        name = file['name']

        # resize
        if "train" in self.set:
            image = image.resize(self.crop_size, Image.BICUBIC)
            label = label.resize(self.crop_size, Image.NEAREST)
        else:
            image = image.resize(self.crop_size, Image.BICUBIC)

        # convert into numpy array
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # remap the semantic label
        label = encode_segmap(label, self.class_mapping, self.ignore_index)

        size = image.shape
        image = image[:, :, ::-1]
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name
