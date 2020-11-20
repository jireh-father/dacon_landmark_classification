import torch
import os
import argparse
import random
import util
import numpy as np
import glob
from PIL import Image
from torch.nn import functional as F

import csv


class CustomDataset(torch.utils.data.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        for image_dir in glob.glob(os.path.join(root, "*")):
            for image_file in glob.glob(os.path.join(image_dir, "*")):
                self.samples.append(image_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path = self.samples[index]
        sample = np.array(Image.open(path).convert("RGB"))
        if self.transform is not None:
            sample = self.transform(image=sample)['image']
        return sample, os.path.splitext(os.path.basename(path))[0]

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_csv_path', type=str, default=None)
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None, required=False, help='checkpoint path')

    parser.add_argument('-t', '--test_dir', type=str)

    parser.add_argument('-m', '--model_name', type=str, default='efficientnet-b1')  # efficientnet-b7', required=False)
    parser.add_argument('--log_step_interval', type=int, default=10)
    parser.add_argument('--pooling', default='GAP', type=str)
    parser.add_argument('--num_classes', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('-w', '--num_workers', type=int, default=8)
    parser.add_argument('--input_size', type=str, default='224')  # 224, 299, 331, 480, 560, 600

    parser.add_argument('--is_different_class_num', action='store_true')
    parser.add_argument('--not_dict_model', action='store_true')

    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('-u', '--use_crop', default=False, action="store_true")
    parser.add_argument('--use_center_crop', default=False, action="store_true")

    parser.add_argument('--center_crop_ratio', default=0.9, type=float)

    parser.add_argument('--use_gray', default=False, action="store_true")

    args = parser.parse_args()

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

    model = util.init_model(args.model_name, num_classes=args.num_classes, pretrained=False,
                            pooling=args.pooling.split(","))
    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        model, optimizer_state, tmp_lr, iteration = util.load_checkpoint(args.checkpoint_path, model,
                                                                         model_name=args.model_name,
                                                                         is_different_class_num=args.is_different_class_num,
                                                                         not_dict_model=args.not_dict_model)
    device = "cuda"
    model = model.to(device)
    model.eval()
    input_size = args.input_size.split(",")
    if len(input_size) == 1:
        input_size = int(input_size[0])
    else:
        input_size = (int(input_size[0]), int(input_size[1]))

    data_loader = util.get_data_loader(CustomDataset, [args.test_dir],
                                       util.get_test_transforms(input_size, args.use_crop,
                                                                center_crop_ratio=args.center_crop_ratio,
                                                                use_gray=args.use_gray),
                                       args.batch_size,
                                       args.num_workers, shuffle=False)

    total_scores = []
    total_indices = []
    total_file_names = []
    for step, (imgs, file_names) in enumerate(data_loader):
        if step > 0 and step % args.log_step_interval == 0:
            print(step, len(data_loader))
        imgs = imgs.to(device)
        total_file_names += list(file_names)
        with torch.set_grad_enabled(False):
            outputs = model(imgs)
            scores, indices = torch.max(F.softmax(outputs, 1), dim=1)
            total_indices += list(indices.cpu().numpy())
            total_scores += list(scores.cpu().numpy())

    rows = zip(total_file_names, total_indices, total_scores)
    output_dir = os.path.dirname(args.output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "landmark_id", "conf"])
        for row in rows:
            writer.writerow(row)
    print("done")
