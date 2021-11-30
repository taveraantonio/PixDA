import os
import numpy as np
import tensorboardX
import logging
from torchvision.transforms.functional import normalize
import torch
import scipy.misc


class Visualizer:

    def __init__(self, args, rank=0, type='tensorboardX', debug=False, filename="logs.txt", summary=True, step=None):
        self.logger = None
        self.args = args
        self.type = type
        self.rank = rank
        self.step = step
        self.summary = summary
        self.denorm = Denormalize(self.args.mean, (1., 1., 1.))

        os.makedirs(os.path.join(self.args.checkpoints_dir, self.args.name), exist_ok=True)
        os.makedirs(os.path.join(self.args.checkpoints_dir, self.args.name, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.args.checkpoints_dir, self.args.name, "logs"), exist_ok=True)
        self.log_dir = os.path.join(self.args.checkpoints_dir, self.args.name, "logs")
        self.filename = os.path.join(self.log_dir, filename)

        if summary:
            if type == 'tensorboardX':
                self.logger = tensorboardX.SummaryWriter(self.log_dir)
            else:
                raise NotImplementedError
        else:
            self.type = 'None'

        self.debug_flag = debug
        logging.basicConfig(filename=self.filename, level=logging.INFO, format=f'%(levelname)s:rank{rank}: %(message)s')

        if rank == 0:
            logging.info(f"[!] starting logging at directory {self.log_dir}")
            if self.debug_flag:
                logging.info(f"[!] Entering DEBUG mode")

    def close(self):
        if self.logger is not None:
            self.logger.close()
        self.info("Closing the Logger.")

    def add_scalar(self, tag, scalar_value, step=None):
        if self.type == 'tensorboardX':
            tag = self._transform_tag(tag)
            self.logger.add_scalar(tag, scalar_value, step)

    def add_image(self, tag, image, step=None):
        if self.type == 'tensorboardX':
            tag = self._transform_tag(tag)
            self.logger.add_image(tag, image, step)

    def add_figure(self, tag, image, step=None):
        if self.type == 'tensorboardX':
            tag = self._transform_tag(tag)
            self.logger.add_figure(tag, image, step)

    def add_table(self, tag, tbl, step=None):
        if self.type == 'tensorboardX':
            tag = self._transform_tag(tag)
            tbl_str = "<table width=\"100%\"> "
            tbl_str += "<tr> \
                     <th>Term</th> \
                     <th>Value</th> \
                     </tr>"
            for k, v in tbl.items():
                tbl_str += "<tr> \
                           <td>%s</td> \
                           <td>%s</td> \
                           </tr>" % (k, v)

            tbl_str += "</table>"
            self.logger.add_text(tag, tbl_str, step)

    def print(self, msg):
        logging.info(msg)

    def info(self, msg):
        if self.rank == 0:
            logging.info(msg)

    def debug(self, msg):
        if self.rank == 0 and self.debug_flag:
            logging.info(msg)

    def error(self, msg):
        logging.error(msg)

    def _transform_tag(self, tag):
        tag = tag + f"/{self.step}" if self.step is not None else tag
        return tag

    def add_results(self, results):
        if self.type == 'tensorboardX':
            tag = self._transform_tag("Results")
            text = "<table width=\"100%\">"
            for k, res in results.items():
                text += f"<tr><td>{k}</td>" + " ".join([str(f'<td>{x}</td>') for x in res.values()]) + "</tr>"
            text += "</table>"
            self.logger.add_text(tag, text)

    def display_current_results(self, visuals, step, mode):
        for k, (img, target, predicted) in enumerate(visuals):
            img = self.denorm(img.detach().cpu().numpy()).astype(np.uint8)
            target = Colorize(self.args.num_classes)(target).detach().cpu().numpy()
            if predicted.size()[0] > 1:
                predicted = predicted.max(0, keepdim=True)[1]
            predicted = Colorize(self.args.num_classes)(predicted[0]).detach().cpu().numpy()
            concat_img = np.concatenate((img, target, predicted), axis=2)  # concat along width
            self.add_image(f'{mode}_{step}', concat_img)

    def save_image(self, img, name):
        os.makedirs(os.path.join(self.args.source_dataroot, "save"), exist_ok=True)
        img = img.detach().cpu().float().numpy()
        img = img.transpose((1, 2, 0))
        scipy.misc.toimage(img, cmin=0.0, cmax=255.0).save(os.path.join(self.args.source_dataroot, "save", name))


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = self.labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

    def uint82bin(self, n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

    def labelcolormap(self, N):
        if N == 35:  # cityscapes
            cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                             (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
                             (102, 102, 156), (190, 153, 153),
                             (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153),
                             (250, 170, 30), (220, 220, 0),
                             (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                             (0, 0, 70),
                             (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32),
                             (0, 0, 142)],
                            dtype=np.uint8)
        elif N == 19:  # Cityscapes remapped to 19 classes
            cmap = np.array([
                (128, 64, 128),  # Road, 0
                (244, 35, 232),  # Sidewalk, 1
                (70, 70, 70),  # Building, 2
                (102, 102, 156),  # Wall, 3
                (190, 153, 153),  # Fence, 4
                (153, 153, 153),  # pole, 5
                (250, 170, 30),  # traffic light, 6
                (220, 220, 0),  # traffic sign, 7
                (107, 142, 35),  # vegetation, 8
                (152, 251, 152),  # terrain, 9
                (70, 130, 180),  # sky, 10
                (220, 20, 60),  # person, 11
                (255, 0, 0),  # rider, 12
                (0, 0, 142),  # car, 13
                (0, 0, 70),  # truck, 14
                (0, 60, 100),  # bus, 15
                (0, 80, 100),  # train, 16
                (0, 0, 230),  # motor-bike, 17
                (119, 11, 32)],  # bike, 18
                dtype=np.uint8)
        else:
            cmap = np.zeros((N, 3), dtype=np.uint8)
            for i in range(N):
                r, g, b = 0, 0, 0
                id = i + 1  # let's give 0 a color
                for j in range(7):
                    str_id = self.uint82bin(id)
                    r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                    g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                    b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                    id = id >> 3
                cmap[i, 0] = r
                cmap[i, 1] = g
                cmap[i, 2] = b

        return cmap
