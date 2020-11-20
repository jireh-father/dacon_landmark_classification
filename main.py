import torch
import torch.nn as nn
import os
import argparse
import random
import util
import trainer
import numpy as np
import glob
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import augmentations
from PIL import Image
import pandas as pd


class CustomDataset(torch.utils.data.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, samples, transform=None):
        self.transform = transform
        self.samples = samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        while True:
            try:
                path, target = self.samples[index]
                sample = np.array(Image.open(path).convert("RGB"))
                if self.transform is not None:
                    sample = self.transform(image=sample)['image']
                return sample, target
            except Exception as e:
                # traceback.print_exc()
                print(str(e), path)
                index = random.randint(0, len(self) - 1)

    def __len__(self):
        return len(self.samples)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=(1, 1)):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets, n_classes, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                         self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class LabelSmoothLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--save_dir', type=str, default=None)
    parser.add_argument('-l', '--log_dir', type=str, default='./log')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None, required=False, help='checkpoint path')

    parser.add_argument('-i', '--train_dir', type=str)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--label_file', type=str)
    # efficientnet-b7
    # resnext101_32x8d
    # resnet152107
    # SPoC', 'MAC', 'GeM'
    parser.add_argument('--pooling', default='GAP', type=str)

    parser.add_argument('-m', '--model_name', type=str, default='efficientnet-b1')  # efficientnet-b7', required=False)
    parser.add_argument('-z', '--optimizer', type=str, default='sgd')  # adam')
    parser.add_argument('--scheduler', type=str, default='cosine')  # cosine, step')
    parser.add_argument('--load_lr', action="store_true", default=False)
    parser.add_argument('--lr_restart_step', type=int, default=1)
    parser.add_argument('-e', '--num_epochs', type=int, default=100)
    parser.add_argument('--log_step_interval', type=int, default=100)

    parser.add_argument('--num_classes', type=int, default=3)

    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('-w', '--num_workers', type=int, default=8)
    parser.add_argument('--input_size', type=str, default='224')  # 224, 299, 331, 480, 560, 600

    parser.add_argument('-p', '--pretrained', default=False, action="store_true")
    parser.add_argument('--is_different_class_num', action='store_true', default=False)
    parser.add_argument('--not_dict_model', action='store_true', default=False)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.9)
    parser.add_argument('-d', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_confusion_matrix', default=False, action="store_true")
    parser.add_argument('-t', '--train', default=False, action="store_true")
    parser.add_argument('-v', '--val', default=False, action="store_true")

    parser.add_argument('-u', '--use_crop', default=False, action="store_true")
    parser.add_argument('--use_center_crop', default=False, action="store_true")
    parser.add_argument('--use_all_train', default=False, action="store_true")
    parser.add_argument('--train_val_data', default=False, action="store_true")
    parser.add_argument('--not_val_shuffle', default=False, action="store_true")
    parser.add_argument('--data_parallel', default=False, action="store_true")

    parser.add_argument('--use_concat_pool', default=False, action="store_true")
    parser.add_argument('--use_no_aug', default=False, action="store_true")
    parser.add_argument('--use_no_color_aug', default=False, action="store_true")
    parser.add_argument('--train_pin_memory', default=False, action="store_true")
    parser.add_argument('--val_pin_memory', default=False, action="store_true")
    parser.add_argument('--label_smoothing', default=False, action="store_true")

    parser.add_argument('--use_benchmark', default=False, action="store_true")

    parser.add_argument('--nesterov', default=False, action="store_true")
    parser.add_argument('--smoothing', default=0.2, type=float)

    parser.add_argument('--cutmix_prob', default=0., type=float)
    parser.add_argument('--beta', default=0., type=float)  # 1.0

    parser.add_argument('--mixup_prob', default=0., type=float)
    parser.add_argument('--alpha', default=0., type=float)  # 1.0

    parser.add_argument('--augmix_prob', default=0., type=float)
    parser.add_argument('--center_crop_ratio', default=0.9, type=float)
    parser.add_argument('--no_jsd', default=False, action="store_true")

    parser.add_argument('--use_gray', default=False, action="store_true")

    parser.add_argument('--class_names', default='normal,warning,chronic,deep', type=str)
    parser.add_argument('--transform_func_name', default='get_train_transforms', type=str)

    # AugMix options
    parser.add_argument(
        '--mixture-width',
        default=3,
        type=int,
        help='Number of augmentation chains to mix per augmented example')
    parser.add_argument(
        '--mixture-depth',
        default=-1,
        type=int,
        help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
    parser.add_argument(
        '--aug-severity',
        default=1,
        type=int,
        help='Severity of base augmentation operators')
    parser.add_argument(
        '--aug-prob-coeff',
        default=1.,
        type=float,
        help='Probability distribution coefficients')
    parser.add_argument(
        '--all-ops',
        '-all',
        action='store_true',
        help='Turn on all operations (+brightness,contrast,color,sharpness).')

    args = parser.parse_args()


    def aug(image, preprocess):
        """Perform AugMix augmentations and compute mixture.
        Args:
          image: PIL.Image input image
          preprocess: Preprocessing function which should return a torch tensor.
        Returns:
          mixed: Augmented and mixed image.
        """
        aug_list = augmentations.augmentations
        if args.all_ops:
            aug_list = augmentations.augmentations_all

        ws = np.float32(
            np.random.dirichlet([args.aug_prob_coeff] * args.mixture_width))
        m = np.float32(np.random.beta(args.aug_prob_coeff, args.aug_prob_coeff))

        mix = torch.zeros_like(preprocess(image=np.array(image))['image'])
        for i in range(args.mixture_width):
            image_aug = image.copy()
            depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                image_aug = op(image_aug, args.aug_severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * preprocess(image=np.array(image_aug))['image']

        mixed = (1 - m) * preprocess(image=np.array(image))['image'] + m * mix
        return mixed


    class AugMixDataset(torch.utils.data.Dataset):
        """Dataset wrapper to perform AugMix augmentation."""

        def __init__(self, samples, preprocess, no_jsd=False):
            self.preprocess = preprocess
            self.no_jsd = no_jsd
            self.samples = samples

        def __getitem__(self, i):
            path, y = self.samples[i]
            x = Image.open(path).convert("RGB")

            # x, y = self.dataset[i]
            if self.no_jsd:
                return aug(x, self.preprocess), y
            else:
                im_tuple = (self.preprocess(image=np.array(x))['image'], aug(x, self.preprocess),
                            aug(x, self.preprocess))
                return im_tuple, y

        def __len__(self):
            return len(self.samples)


    print("training params")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    if args.seed:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        if args.use_benchmark:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True

    pretrained = True if args.pretrained and args.checkpoint_path is None else False
    model = util.init_model(args.model_name, num_classes=args.num_classes, pretrained=pretrained,
                            pooling=args.pooling.split(","))
    lr = args.lr
    iteration = 0
    optimizer_state = None
    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        model, optimizer_state, tmp_lr, tmp_iteration = util.load_checkpoint(args.checkpoint_path, model,
                                                                             model_name=args.model_name,
                                                                             is_different_class_num=args.is_different_class_num,
                                                                             not_dict_model=args.not_dict_model)
        if args.load_lr and not args.not_dict_model:
            lr = tmp_lr
            # iteration = tmp_iteration
            print("loaded lr", lr)
    if torch.cuda.device_count() > 1 and args.data_parallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    device = "cuda"

    if args.use_concat_pool and args.model_name.startswith("efficientnet"):
        print("apply AdaptiveConcatPool2d")
        model._avg_pooling = AdaptiveConcatPool2d(1)
        model._fc = nn.Linear(model._fc.in_features * 2, model._fc.out_features)
        print(model)

    model = model.to(device)

    phases = []
    data_loaders = {}

    train_file_map = {}
    for parent_dir in glob.glob(os.path.join(args.train_dir, "*")):
        for sub_dir in glob.glob(os.path.join(glob.escape(parent_dir), "*")):
            for file_path in glob.glob(os.path.join(glob.escape(sub_dir), "*")):
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                train_file_map[file_name] = file_path
    label_df = pd.read_csv(args.label_file)
    landmark_ids = label_df.landmark_id.unique()
    val_items = []
    train_items = []
    for landmark_id in landmark_ids:
        tmp_df = label_df[label_df.landmark_id == landmark_id]
        if len(tmp_df) < 1:
            print("OMG")
            import sys

            sys.exit()
        val_cnt = round(len(tmp_df) * args.val_ratio)
        # print(tmp_df.values[0], "train: {}, val: {}".format(len(tmp_df) - val_cnt, val_cnt))
        items = list(zip(tmp_df.id.values, tmp_df.landmark_id.values))
        if not args.not_val_shuffle:
            random.shuffle(items)
        tmp_val_items = [(train_file_map[k], l) for k, l in items[:val_cnt]]
        val_items += tmp_val_items
        if args.train_val_data:
            train_items += tmp_val_items
        else:
            if args.use_all_train:
                train_items += [(train_file_map[k], l) for k, l in items]
            else:
                train_items += [(train_file_map[k], l) for k, l in items[val_cnt:]]
    print("val_items", len(val_items))

    if args.train:
        train_dirs = args.train_dir.split(",")
        label_count = None
        for train_dir in train_dirs:
            class_dirs = glob.glob(os.path.join(train_dir, "*"))
            class_dirs.sort()
            if label_count is None:
                label_count = np.array([len(glob.glob(os.path.join(class_dir, "*"))) for class_dir in class_dirs])
            else:
                label_count += np.array([len(glob.glob(os.path.join(class_dir, "*"))) for class_dir in class_dirs])
        label_count = list(label_count)
        max_count = max(label_count)
        # weights = [max_count / cnt for cnt in label_count]
        # weights = [math.log10(max_count / cnt) + 1 for cnt in label_count]
        # weights = [math.log2(max_count / cnt) + 1 for cnt in label_count]

        if args.label_smoothing:
            # criterion = LabelSmoothingLoss(args.num_classes, smoothing=args.smoothing)
            # criterion = LabelSmoothLoss(smoothing=args.smoothing)
            criterion = LabelSmoothingCrossEntropy(epsilon=args.smoothing)
            # criterion = SmoothCrossEntropyLoss(weight=class_weights,smoothing=args.smoothing)
        else:
            # weights = [math.log(max_count / cnt) + 1 for cnt in label_count]
            # print("class_weights", weights)
            # class_weights = torch.FloatTensor(weights).to(device)
            # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            criterion = torch.nn.CrossEntropyLoss()  # weight=class_weights)
        if not args.load_lr:
            optimizer_state = None

        steps_per_epoch = len(train_items) // args.train_batch_size
        if len(train_items) % args.train_batch_size > 0:
            steps_per_epoch += 1
        optimizer_ft, exp_lr_scheduler, use_lr_schedule_steps = util.init_optimizer(args.optimizer, model,
                                                                                    optimizer_state,
                                                                                    lr, args.weight_decay,
                                                                                    args.lr_restart_step,
                                                                                    args.lr_decay_gamma,
                                                                                    args.scheduler,
                                                                                    nesterov=args.nesterov,
                                                                                    num_epochs=args.num_epochs,
                                                                                    steps_per_epoch=steps_per_epoch)

    else:
        optimizer_ft = None
        exp_lr_scheduler = None
        use_lr_schedule_steps = False
        criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    input_size = args.input_size.split(",")
    if len(input_size) == 1:
        input_size = int(input_size[0])
    else:
        input_size = (int(input_size[0]), int(input_size[1]))

    if args.train:
        # train_dataset = CustomDataset(args.train_dir,
        #                               transform=util.get_train_transforms(input_size, args.use_crop))
        # train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
        #                                                 shuffle=True, num_workers=args.num_workers, pin_memory=True)

        print("started to create train data loader")
        if args.augmix_prob > 0:
            train_data_loader = util.get_data_loader_aug_mix(AugMixDataset,
                                                             util.get_preprocess(input_size, args.use_crop,
                                                                                 args.use_center_crop,
                                                                                 args.center_crop_ratio,
                                                                                 use_gray=args.use_gray),
                                                             [train_items], args.train_batch_size,
                                                             args.num_workers, shuffle=True,
                                                             pin_memory=args.train_pin_memory, no_jsd=args.no_jsd, )
        else:
            if args.use_no_aug:
                train_transforms = util.get_preprocess(input_size, args.use_crop, args.use_center_crop,
                                                       args.center_crop_ratio,
                                                       use_gray=args.use_gray)
            else:
                transform_func = getattr(util, args.transform_func_name)
                train_transforms = transform_func(input_size, args.use_crop, args.use_no_color_aug,
                                                  use_center_crop=args.use_center_crop,
                                                  center_crop_ratio=args.center_crop_ratio,
                                                  use_gray=args.use_gray)
            train_data_loader = util.get_data_loader(CustomDataset, [train_items],
                                                     train_transforms,
                                                     args.train_batch_size,
                                                     args.num_workers, shuffle=True, pin_memory=args.train_pin_memory)

        data_loaders['train'] = train_data_loader
        phases.append("train")
        print("created train data loader")

    if args.val:
        val_data_loader = util.get_data_loader(CustomDataset, [val_items],
                                               util.get_test_transforms(input_size, args.use_crop,
                                                                        center_crop_ratio=args.center_crop_ratio,
                                                                        use_gray=args.use_gray),
                                               args.val_batch_size,
                                               args.num_workers, shuffle=False, pin_memory=args.val_pin_memory)
        print("created val data loader")
        data_loaders['val'] = {'val': val_data_loader}
        phases.append("val")

    class_names = args.class_names.split(",") if args.class_names else None
    trainer.train_or_val(model, criterion, optimizer_ft, exp_lr_scheduler, data_loaders, num_epochs=args.num_epochs,
                         log_step_interval=args.log_step_interval, device=device,
                         train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size,
                         save_dir=args.save_dir,
                         start_step=iteration,
                         log_dir=args.log_dir,
                         use_lr_schedule_steps=use_lr_schedule_steps,
                         save_confusion_matrix=args.save_confusion_matrix,
                         phases=phases, class_names=class_names,
                         cutmix_prob=args.cutmix_prob, beta=args.beta,
                         mixup_prob=args.mixup_prob, alpha=args.alpha,
                         augmix_prob=args.augmix_prob, no_jsd=args.no_jsd, model_name=args.model_name)
