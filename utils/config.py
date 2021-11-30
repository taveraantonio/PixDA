import argparse
import pickle
import os
import utils.utils as util
import numpy as np


class Options:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # experiment specifics
        parser.add_argument('--is_train', action='store_true', help='True if training, False otherwise')
        parser.add_argument('--name', type=str, default='GTA5toCityscapes_1shot', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2.')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='deeplabv2', help='which model to use, e.g. (deeplabv2|deeplabv3)')
        parser.add_argument('--multi_level', type=bool, default=True, help='extract features at multiple levels of the model')
        parser.add_argument('--restore_from', type=str, default='./models/pretrained_models/DeepLab_resnet_pretrained_init.pth')
        parser.add_argument('--ft', type=str, default=None, help='Restore model path to finetune on')
        parser.add_argument('--num_shots', default=1, type=int, help='# shot for the experiment')

        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_iter', type=int, default=6784, help='which iter to load when resuming')
        parser.add_argument('--max_iters', type=int, default=250000, help='# of max steps to do during training')
        parser.add_argument('--early_stop', type=int, default=120000, help='# early stop iteration')
        parser.add_argument('--max_epochs', type=int, default=6, help='# maximum training epochs')
        parser.add_argument('--seed', type=int, default=1993, help='seed to set random seed')

        # for displays/saving/validating
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')

        # input/output sizes
        parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
        parser.add_argument('--batch_size_val', type=int, default=2, help='validation batch size')
        parser.add_argument('--crop_size', type=tuple, default=(1024, 512), help='Crop to the width of crop_size')
        parser.add_argument('--num_workers', default=8, type=int, help='#threads for loading data')

        # for dataset
        parser.add_argument('--num_classes', type=int, default=19, help='# of input label classes')
        parser.add_argument('--ignore_index', default=255, type=int, help='ignore index')
        parser.add_argument('--source_dataset', type=str, default='gta5', help='source dataset name')
        parser.add_argument('--target_dataset', type=str, default='cityscapes', help='target dataset name')
        parser.add_argument('--source_dataroot', type=str, default='./datasets/GTA5')
        parser.add_argument('--target_dataroot', type=str, default='./datasets/Cityscapes')
        parser.add_argument('--mean', type=int, default=(104.00698793, 116.66876762, 122.67891434), help='dataset mean after style translation')
        parser.add_argument('--mean_pre', type=int, default=(0.0, 0.0, 0.0), help='dataset mean before style translation')

        # for sample selection
        parser.add_argument('--ss_threshold', type=float, default=0.4, help="starting threshold for sample selection")
        parser.add_argument('--inc', type=float, default=0.4, help="value to increment threshold at each epoch")

        # for segmentation
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum optimizer')
        parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay optimizer')
        parser.add_argument('--seg_lr', type=float, default=2.5e-4, help='learning rate for scene parser')
        parser.add_argument('--power', type=float, default=0.9, help='poly learning rate decay power')
        parser.add_argument('--seg_norm', type=str, default=None, help='(instance_norm|None) if None batch_norm as default')
        parser.add_argument('--seg_loss', type=str, default='focal', help='(focal|None) if None it uses cross entropy')

        # for distillation
        parser.add_argument('--alpha', type=float, default=0.5, help="distillation parameter")
        parser.add_argument('--lambda_kd', type=float, default=0.5, help="lambda value for distillation loss")
        parser.add_argument('--kd_steps', type=int, default=201, help='# of steps to fine-tune the pixda model')

        # for discriminators
        parser.add_argument('--lambda_adv', type=float, default=0.1, help="lambda adv for discriminator 0.1")
        parser.add_argument('--d_lr', type=float, default=1e-4, help='learning rate for discriminators')
        parser.add_argument('--beta1_d', type=float, default=0.9, help='momentum term for adam')
        parser.add_argument('--beta2_d', type=float, default=0.99, help='momentum term for adam')

        self.initialized = True
        return parser

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def parse(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        opt = parser.parse_args()
        self.parser = parser

        # adapt iterations and learning rates to batch size value
        opt.max_iters = int(opt.max_iters / opt.batch_size)
        opt.early_stop = int(opt.early_stop / opt.batch_size)
        opt.seg_lr = opt.seg_lr * opt.batch_size
        opt.d_lr = opt.d_lr * opt.batch_size

        opt.th = str(opt.threshold).replace(".", "")

        # print and save options
        self.print_options(opt)
        if opt.is_train:
            self.save_options(opt)

        # reformat gpu_ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        # convert mean to array
        opt.mean = np.array(opt.mean, dtype=np.float32)

        assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, \
            "Training batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batch_size, len(opt.gpu_ids))

        assert len(opt.gpu_ids) == 0 or opt.batch_size_val % len(opt.gpu_ids) == 0, \
            "Validation batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batch_size_val, len(opt.gpu_ids))

        self.opt = opt
        return opt
