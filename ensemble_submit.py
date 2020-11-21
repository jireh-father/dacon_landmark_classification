import torch
import os
import argparse
import random
import util
import numpy as np
import glob
from PIL import Image
from torch.nn import functional as F
import sys
import csv
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import pickle


class EvalDataset(torch.utils.data.Dataset):
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


class TestDataset(torch.utils.data.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        for image_dir in glob.glob(os.path.join(root, "*")):
            for image_file in glob.glob(os.path.join(image_dir, "*")):
                self.samples.append(image_file)
        self.samples.sort()

        # self.samples.sort()

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


def get_val_samples(image_dir, label_file, val_ratio):
    train_file_map = {}
    for parent_dir in glob.glob(os.path.join(image_dir, "*")):
        for sub_dir in glob.glob(os.path.join(glob.escape(parent_dir), "*")):
            for file_path in glob.glob(os.path.join(glob.escape(sub_dir), "*")):
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                train_file_map[file_name] = file_path
    label_df = pd.read_csv(label_file)
    landmark_ids = label_df.landmark_id.unique()
    val_items = []
    for landmark_id in landmark_ids:
        tmp_df = label_df[label_df.landmark_id == landmark_id]
        if len(tmp_df) < 1:
            print("OMG")
            import sys

            sys.exit()
        val_cnt = round(len(tmp_df) * val_ratio)
        # print(tmp_df.values[0], "train: {}, val: {}".format(len(tmp_df) - val_cnt, val_cnt))
        items = list(zip(tmp_df.id.values, tmp_df.landmark_id.values))
        random.shuffle(items)
        val_items += [(train_file_map[k], l) for k, l in items[:val_cnt]]
    return val_items


def main(args):
    if args.use_glob:
        checkpoint_paths = glob.glob(args.checkpoint_paths)
        checkpoint_paths.sort()
    else:
        checkpoint_paths = args.checkpoint_paths.split(",")

    print("all checkpoints", checkpoint_paths)

    if args.weights is None:
        weights = [1.0] * len(checkpoint_paths)
    else:
        weights = [float(w) for w in args.weights.split(",")]

        if not args.use_glob and len(checkpoint_paths) != len(weights):
            sys.exit("weights count not matched with checkpoint_paths")

    if not args.from_pkl:

        model_names = args.model_names.split(":")

        if len(model_names) == 1:
            model_names = model_names * len(checkpoint_paths)

        if len(checkpoint_paths) != len(model_names):
            sys.exit("model_names count not matched with checkpoint_paths")

        input_sizes = args.input_sizes.split(",")
        if len(input_sizes) == 1:
            input_sizes = input_sizes * len(checkpoint_paths)

        if len(checkpoint_paths) != len(input_sizes):
            sys.exit("input_sizes count not matched with checkpoint_paths")

        new_input_sizes = []
        for input_size in input_sizes:
            input_size = input_size.split("x")
            if len(input_size) == 1:
                input_size = int(input_size[0])
            else:
                input_size = (int(input_size[0]), int(input_size[1]))
            new_input_sizes.append(input_size)
        input_sizes = new_input_sizes

        if args.use_crops:
            use_crops = args.use_crops.split(",")
            if len(use_crops) == 1:
                use_crops = use_crops * len(checkpoint_paths)

            if len(checkpoint_paths) != len(use_crops):
                sys.exit("use_crops count not matched with checkpoint_paths")

            use_crops = [use_crop in ['true', 'True'] for use_crop in use_crops]
        else:
            use_crops = [False] * len(checkpoint_paths)

        if args.poolings:
            poolings = args.poolings.split(":")
            if len(poolings) == 1:
                poolings = [poolings[0].split(",")]
                poolings = poolings * len(checkpoint_paths)
            else:
                if len(checkpoint_paths) != len(poolings):
                    sys.exit("poolings count not matched with checkpoint_paths")

                poolings = [pooling.split(",") for pooling in poolings]
        else:
            poolings = [['GAP']] * len(checkpoint_paths)

        if args.eval or args.save:
            val_samples = get_val_samples(args.image_dir, args.label_file, args.val_ratio)
            val_samples.sort()
            print("val samples", len(val_samples))
        eval_logits = None
        test_logits = None

        if args.save:
            os.makedirs(args.output_dir, exist_ok=True)

        for i, checkpoint_path in enumerate(checkpoint_paths):
            print(checkpoint_path, i, len(checkpoint_paths))
            model = util.init_model(model_names[i], num_classes=args.num_classes, pretrained=False, pooling=poolings[i])
            model, _, _, _ = util.load_checkpoint(checkpoint_path, model,
                                                  model_name=model_names[i],
                                                  is_different_class_num=False,
                                                  not_dict_model=False,
                                                  strict=not args.not_strict)
            device = "cuda"
            model = model.to(device)
            model.eval()
            input_size = input_sizes[i]
            print("input size", input_size)
            if args.eval or args.save:
                data_loader = util.get_data_loader(EvalDataset, [val_samples],
                                                   util.get_test_transforms(input_size, use_crops[i],
                                                                            center_crop_ratio=args.center_crop_ratio,
                                                                            use_gray=args.use_gray),
                                                   args.batch_size,
                                                   args.num_workers, shuffle=False)

                epoch_labels = []
                epoch_preds = []
                total_logits = None
                for step, (inputs, labels) in enumerate(data_loader):
                    epoch_labels += list(labels.numpy())
                    inputs = inputs.to(device)

                    with torch.set_grad_enabled(False):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        epoch_preds += list(preds.cpu().numpy())
                        logits = F.softmax(outputs, 1)
                        if total_logits is None:
                            total_logits = logits.cpu().numpy()
                        else:
                            total_logits = np.concatenate([total_logits, logits.cpu().numpy()], axis=0)

                        _, preds = torch.max(outputs, 1)

                    if step > 0 and step % args.log_step_interval == 0:
                        print("evaluating", step, len(data_loader))
                tmp_acc = accuracy_score(epoch_labels, epoch_preds)
                tmp_f1 = f1_score(epoch_labels, epoch_preds, average='macro')
                print("acc: {}, f1: {}".format(tmp_acc, tmp_f1))

                if args.save:
                    pickle.dump({"logits": total_logits, "labels": epoch_labels}, open(
                        os.path.join(args.output_dir, os.path.basename(os.path.dirname(checkpoint_path)) + "_" +
                                     os.path.splitext(os.path.basename(checkpoint_path))[0] + "_eval.pkl"),
                        "wb+"))

                total_logits *= weights[i]
                if eval_logits is None:
                    eval_logits = total_logits
                else:
                    eval_logits += total_logits

            if args.test or args.save:
                data_loader = util.get_data_loader(TestDataset, [args.test_dir],
                                                   util.get_test_transforms(input_size, use_crops[i],
                                                                            center_crop_ratio=args.center_crop_ratio,
                                                                            use_gray=args.use_gray,
                                                                            use_pad=args.use_pad),
                                                   args.batch_size,
                                                   args.num_workers, shuffle=False)

                total_file_names = []
                total_logits = None
                for step, (imgs, file_names) in enumerate(data_loader):
                    if step > 0 and step % args.log_step_interval == 0:
                        print("testing", step, len(data_loader))
                    imgs = imgs.to(device)
                    total_file_names += list(file_names)
                    with torch.set_grad_enabled(False):
                        outputs = model(imgs)
                        logits = F.softmax(outputs, 1)
                        if total_logits is None:
                            total_logits = logits.cpu().numpy()
                        else:
                            total_logits = np.concatenate([total_logits, logits.cpu().numpy()], axis=0)

                if args.save:
                    pickle.dump({"logits": total_logits, "file_names": total_file_names}, open(
                        os.path.join(args.output_dir, os.path.basename(os.path.dirname(checkpoint_path)) + "_" +
                                     os.path.splitext(os.path.basename(checkpoint_path))[0] + "_test.pkl"),
                        "wb+"))
                total_logits *= weights[i]
                if test_logits is None:
                    test_logits = total_logits
                else:
                    test_logits += total_logits

            del model

    if args.eval:
        if args.from_pkl:
            eval_logits = None
            if args.use_glob:
                checkpoint_paths = [os.path.basename(p)[:-9] for p in
                                    glob.glob(os.path.join(args.pkl_dir, "*_eval.pkl"))]
                weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)
                print(weights)

            for i, logits_file in enumerate(checkpoint_paths):
                print(logits_file)
                logits = pickle.load(open(os.path.join(args.pkl_dir, logits_file + "_eval.pkl"), "rb+"))
                epoch_labels = logits['labels']
                logits = logits['logits']
                if eval_logits is None:
                    eval_logits = logits * weights[i]
                else:
                    eval_logits += logits * weights[i]
        else:
            eval_logits = eval_logits  # / len(checkpoint_paths)
        eval_indices = []
        for eval_logit in eval_logits:
            index = np.argmax(eval_logit)
            eval_indices.append(index)

        acc = accuracy_score(epoch_labels, eval_indices)
        f1 = f1_score(epoch_labels, eval_indices, average='macro')
        print("last acc", acc)
        print("last fq", f1)

    if args.test:
        if args.from_pkl:
            test_logits = None
            if args.use_glob:
                checkpoint_paths = [os.path.basename(p)[:-9] for p in
                                    glob.glob(os.path.join(args.pkl_dir, "*_eval.pkl"))]
                weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)
                print(weights)
            for i, logits_file in enumerate(checkpoint_paths):
                logits = pickle.load(open(os.path.join(args.pkl_dir, logits_file + "_test.pkl"), "rb+"))
                total_file_names = logits['file_names']
                logits = logits['logits']
                if test_logits is None:
                    test_logits = logits * weights[i]
                else:
                    test_logits += logits * weights[i]
        else:
            test_logits = test_logits  # / len(checkpoint_paths)

        if sum(weights) != 1:
            test_logits += (1 - sum(weights))
        test_indices = []
        test_scores = []
        for test_logit in test_logits:
            index = np.argmax(test_logit)
            score = test_logit[index]
            test_indices.append(index)
            test_scores.append(score)

        rows = zip(total_file_names, test_indices, test_scores)
        output_dir = os.path.dirname(args.csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.csv, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "landmark_id", "conf"])
            for row in rows:
                writer.writerow(row)
        print("test done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--pkl_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('-c', '--checkpoint_paths', type=str, default=None, required=False, help='checkpoint path')
    parser.add_argument('--use_glob', action='store_true', default=False)

    parser.add_argument('-i', '--image_dir', type=str)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--label_file', type=str)
    parser.add_argument('-t', '--test_dir', type=str)

    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('-m', '--model_names', type=str, default='efficientnet-b1')  # efficientnet-b7', required=False)
    parser.add_argument('--log_step_interval', type=int, default=10)
    parser.add_argument('--poolings', default='GAP', type=str)
    parser.add_argument('--num_classes', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('-w', '--num_workers', type=int, default=8)
    parser.add_argument('--input_sizes', type=str, default='224')  # 224, 299, 331, 480, 560, 600

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--from_pkl', action='store_true', default=False)
    parser.add_argument('--use_pad', action='store_true', default=False)

    parser.add_argument('--not_strict', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('-u', '--use_crops', default=None, type=str)
    parser.add_argument('--use_center_crop', default=False, action="store_true")

    parser.add_argument('--center_crop_ratio', default=0.9, type=float)

    parser.add_argument('--use_gray', default=None, type=str)

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

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    main(args)
