import torch
import time
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision
import util
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import itertools
from torch.autograd import Variable
import torch.nn.functional as F


def train_or_val(model, criterion, optimizer, scheduler, data_loaders, num_epochs=25, log_step_interval=100,
                 device=None, train_batch_size=64,
                 val_batch_size=128, save_dir=None, start_step=0, log_dir=None, use_lr_schedule_steps=False,
                 save_confusion_matrix=True, phases=['train', 'val'], class_names=None, cutmix_prob=0., beta=0,
                 mixup_prob=0., alpha=0.,
                 augmix_prob=0., no_jsd=False, model_name=None):
    print("Start training or validating!")
    since = time.time()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'val'))
    val_bests = {}
    val_writers = {}
    for val_name in data_loaders['val']:
        val_bests[val_name] = {'best_f1': 0.0, 'best_acc': 0.0, 'best_epoch': 1}
        val_writers[val_name] = SummaryWriter(os.path.join(log_dir, 'val_{}'.format(val_name)))

    start_time = datetime.datetime.now()

    if 'train' in data_loaders:
        start_epoch = start_step // len(data_loaders['train']) + 1
    else:
        start_epoch = 1

    for epoch in range(start_epoch, num_epochs + 1):
        if 'train' in phases:
            train_epoch(model, optimizer, criterion, data_loaders['train'], epoch, num_epochs, scheduler,
                        use_lr_schedule_steps,
                        train_batch_size, device, log_step_interval, train_writer, save_dir, start_time,
                        cutmix_prob=cutmix_prob, beta=beta, mixup_prob=mixup_prob, alpha=alpha,
                        augmix_prob=augmix_prob, no_jsd=no_jsd, model_name=model_name)

        if 'val' in phases:
            epoch_labels = []
            epoch_preds = []
            epoch_losses = 0.
            epoch_corrects = 0
            epoch_samples = 0
            for val_name in data_loaders['val']:
                validation_result = validate(model, criterion, data_loaders['val'][val_name],
                                             val_batch_size, epoch,
                                             device, val_writers[val_name], val_name, log_dir,
                                             log_step_interval,
                                             save_confusion_matrix, start_time,
                                             best_f1=val_bests[val_name]['best_f1'],
                                             best_acc=val_bests[val_name]['best_acc'],
                                             best_epoch=val_bests[val_name]['best_epoch'],
                                             class_names=class_names)

                best_f1, best_acc, best_epoch, is_best, tmp_epoch_labels, tmp_epoch_preds, running_loss, running_corrects, samples = validation_result
                epoch_labels += tmp_epoch_labels
                epoch_preds += tmp_epoch_preds
                epoch_losses += running_loss
                epoch_corrects += running_corrects
                epoch_samples += samples
                if 'train' in phases and is_best:
                    val_bests[val_name]['best_f1'] = best_f1
                    val_bests[val_name]['best_acc'] = best_acc
                    val_bests[val_name]['best_epoch'] = best_epoch
                    util.save_checkpoint(model, optimizer, scheduler.get_lr()[0], len(data_loaders['train']) * epoch,
                                         os.path.join(save_dir, "{}_best.pth".format(val_name)))

            epoch_loss = epoch_losses / epoch_samples
            epoch_acc = float(epoch_corrects) / epoch_samples
            f1 = f1_score(epoch_labels, epoch_preds, average='macro')
            if save_confusion_matrix:
                cls_report = classification_report(epoch_labels, epoch_preds)  # , target_names=classes)
                print("Total val classification_report")
                print(cls_report)
                with open(os.path.join(log_dir, "total_eval_report.txt"), "a+") as fp:
                    fp.write("[epoch:%02d] loss: %f, acc: %f\n" % (epoch, epoch_loss, epoch_acc))
                    epoch_cm = confusion_matrix(epoch_labels, epoch_preds)
                    np_epoch_labels = np.unique(np.array(epoch_labels))
                    np_epoch_labels.sort()
                    log_confusion_matrix(val_writer, epoch, epoch_cm, np_epoch_labels)
                    fp.write(str(epoch_cm) + "\n")
                    print("confusion matrix")
                    print(epoch_cm)
                    epoch_cm = epoch_cm.astype('float') / epoch_cm.sum(axis=1)[:, np.newaxis]
                    epoch_cm = epoch_cm.diagonal()
                    fp.write(str(epoch_cm) + "\n")
                    print("each accuracies")
                    print(epoch_cm)
                    fp.write(str(cls_report) + "\n")

            val_writer.add_scalar('Loss/epoch', epoch_loss, epoch)
            val_writer.add_scalar('Acc/epoch', epoch_acc, epoch)
            val_writer.add_scalar('F1/epoch', f1, epoch)

            if 'train' not in phases:
                print("The end of validation.")
                break

    time_elapsed = time.time() - since
    print('Training or Validation completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    if 'val' in phases:
        for val_name in val_bests:
            print('%s Val - Best F1: %f, Best Acc: %f, Best Epoch %d' % (val_name, val_bests[val_name]['best_f1'],
                                                                         val_bests[val_name]['best_acc'],
                                                                         val_bests[val_name]['best_epoch']))
    train_writer.close()
    val_writer.close()
    for key in val_writers:
        val_writers[key].close()


def train_epoch(model, optimizer, criterion, data_loader, epoch, num_epochs, scheduler, use_lr_schedule_steps,
                batch_size, device, log_step_interval, writer, save_dir, start_time, cutmix_prob=0., beta=0.,
                mixup_prob=0., alpha=0., augmix_prob=0., no_jsd=False, model_name=None):
    total_train_steps = len(data_loader)

    print("Epoch %d/%d, LR: %f" % (epoch, num_epochs, np.array(scheduler.get_lr()).mean()))
    # print("LR", scheduler.get_lr())

    print('-' * 10)

    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0
    multi_batch_loss = 0.
    multi_batch_corrects = 0
    epoch_elapsed = 0.

    # Iterate over data.
    multi_batch_elapsed_time = 0.
    epoch_preds = []
    epoch_labels = []
    multi_batch_preds = []
    multi_batch_labels = []
    for step, (inputs, labels) in enumerate(data_loader):
        batch_start_time = time.time()
        epoch_labels += list(labels.numpy())
        multi_batch_labels += list(labels.numpy())
        if augmix_prob <= 0 or no_jsd:
            inputs = inputs.to(device)
        labels = labels.to(device)

        if use_lr_schedule_steps:
            scheduler.step(epoch - 1 + step / len(data_loader))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            r = np.random.rand(1)
            if isinstance(inputs, list):
                inputs_size = inputs[0].size(0)
            else:
                inputs_size = inputs.size(0)
            if beta > 0 and r < cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                # compute output
                if model_name.startswith("arc_face"):
                    outputs = model(inputs, labels)
                    loss1 = criterion(outputs[0], target_a) * lam + criterion(outputs[0], target_b) * (1. - lam)
                    loss2 = criterion(outputs[1], target_a) * lam + criterion(outputs[0], target_b) * (1. - lam)
                    loss = loss1 * 0.2 + loss2 * 0.8
                    outputs = outputs[1]
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
            elif r < mixup_prob:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha, device)
                inputs, targets_a, targets_b = map(Variable, (inputs,
                                                              targets_a, targets_b))

                if model_name.startswith("arc_face"):
                    outputs = model(inputs, labels)
                    loss1 = mixup_criterion(criterion, outputs[0], targets_a, targets_b, lam) * 0.2
                    loss2 = mixup_criterion(criterion, outputs[1], targets_a, targets_b, lam) * 0.8
                    loss = loss1 + loss2
                    outputs = outputs[1]
                else:
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            elif augmix_prob > 0:
                if no_jsd:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                else:

                    images_all = torch.cat(inputs, 0).to(device)
                    outputs = model(images_all)
                    logits_clean, logits_aug1, logits_aug2 = torch.split(
                        outputs, inputs[0].size(0))
                    outputs = logits_clean
                    # Cross-entropy is only computed on clean images
                    loss = criterion(logits_clean, labels)

                    p_clean, p_aug1, p_aug2 = F.softmax(
                        logits_clean, dim=1), F.softmax(
                        logits_aug1, dim=1), F.softmax(
                        logits_aug2, dim=1)

                    # Clamp mixture distribution to avoid exploding KL divergence
                    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                    loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                                  F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                                  F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            else:

                if model_name.startswith("arc_face"):
                    outputs = model(inputs, labels)
                    loss1 = criterion(outputs[0], labels) * 0.2
                    loss2 = criterion(outputs[1], labels) * 0.8
                    loss = loss1 + loss2
                    outputs = outputs[1]
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            epoch_preds += list(preds.cpu().numpy())
            multi_batch_preds += list(preds.cpu().numpy())

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs_size
        running_corrects += int(torch.sum(preds == labels.data).cpu().numpy())
        multi_batch_loss += loss.item()
        multi_batch_corrects += int(torch.sum(preds == labels.data).cpu().numpy())
        batch_elapsed_time = time.time() - batch_start_time
        multi_batch_elapsed_time += batch_elapsed_time
        epoch_elapsed += batch_elapsed_time

        if step >= 0 and (step + 1) % log_step_interval == 0:
            current_datetime = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            elapsed_time = datetime.datetime.now() - start_time

            multi_batch_f1 = f1_score(multi_batch_labels, multi_batch_preds, average='macro')

            if isinstance(inputs, list):
                grid = torchvision.utils.make_grid(inputs[0])
                grid = util.denormalize_image(grid)
                writer.add_image('images1', grid, step + (total_train_steps * (epoch - 1)),
                                 dataformats='HWC')
                grid = torchvision.utils.make_grid(inputs[1])
                grid = util.denormalize_image(grid)
                writer.add_image('images2', grid, step + (total_train_steps * (epoch - 1)),
                                 dataformats='HWC')
            else:
                grid = torchvision.utils.make_grid(inputs)
                grid = util.denormalize_image(grid)
                writer.add_image('images', grid, step + (total_train_steps * (epoch - 1)),
                                 dataformats='HWC')
            writer.add_scalar('Loss', multi_batch_loss / log_step_interval,
                              step + (total_train_steps * (epoch - 1)))
            writer.add_scalar('Acc',
                              multi_batch_corrects / (log_step_interval * batch_size),
                              step + (total_train_steps * (epoch - 1)))
            writer.add_scalar('F1', multi_batch_f1, step + (total_train_steps * (epoch - 1)))
            writer.add_scalar('LR', np.array(scheduler.get_lr()).mean(),
                              step + (total_train_steps * (epoch - 1)))

            print(
                "[train-epoch:%02d/%02d,step:%d/%d,%s] total_elapsed: %s, batch_elapsed: %f, %d steps_elapsed: %f"
                % (epoch, num_epochs, step, len(data_loader),
                   current_datetime, elapsed_time, batch_elapsed_time, log_step_interval,
                   multi_batch_elapsed_time))
            print("loss: %f, acc: %f, lr: %f, multi_batch_loss: %f, multi_batch_acc: %f, multi_batch_f1: %f" % (
                loss.item(), float(torch.sum(preds == labels.data).cpu().numpy()) / batch_size,
                np.array(scheduler.get_lr()).mean(), multi_batch_loss / log_step_interval,
                multi_batch_corrects / (log_step_interval * batch_size), multi_batch_f1))
            # print("LR", scheduler.get_lr())
            multi_batch_elapsed_time = 0.
            multi_batch_loss = 0.
            multi_batch_corrects = 0
            multi_batch_preds = []
            multi_batch_labels = []

    if not use_lr_schedule_steps:
        scheduler.step()

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = float(running_corrects) / len(data_loader.dataset)

    elapsed_time = datetime.datetime.now() - start_time
    current_datetime = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    f1 = f1_score(epoch_labels, epoch_preds, average='macro')

    print(
        "[result:train-epoch:%02d/%02d,%s] total_elapsed: %s, epoch_elapsed: %s, loss: %f, acc: %f, f1: %f, lr: %f" % (
            epoch, num_epochs, current_datetime, elapsed_time, epoch_elapsed, epoch_loss, epoch_acc, f1,
            scheduler.get_lr()[0]))
    writer.add_scalar('Loss/epoch', epoch_loss, epoch)
    writer.add_scalar('Acc/epoch', epoch_acc, epoch)
    writer.add_scalar('F1/epoch', f1, epoch)
    util.save_checkpoint(model, optimizer, scheduler.get_lr()[0], total_train_steps * epoch,
                         os.path.join(save_dir, "epoch_%d.pth" % epoch))

    print()


def validate(model, criterion, data_loader, batch_size, epoch, device, writer, val_name, log_dir, log_step_interval,
             save_confusion_matrix, start_time, best_f1=None, best_acc=None, best_epoch=None, class_names=None):
    # Each epoch has a training and validation phase
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    multi_batch_loss = 0.
    multi_batch_corrects = 0
    epoch_elapsed = 0.

    # Iterate over data.
    multi_batch_elapsed_time = 0.
    epoch_preds = []
    epoch_labels = []

    print("started to validate {} val dataset. samples: {}".format(val_name, len(data_loader.dataset)))
    for step, (inputs, labels) in enumerate(data_loader):
        batch_start_time = time.time()
        epoch_labels += list(labels.numpy())
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            start = time.time()
            outputs = model(inputs)
            # print("batch speed", time.time() - start)
            _, preds = torch.max(outputs, 1)
            epoch_preds += list(preds.cpu().numpy())
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += int(torch.sum(preds == labels.data).cpu().numpy())
        multi_batch_loss += loss.item()
        multi_batch_corrects += int(torch.sum(preds == labels.data).cpu().numpy())
        batch_elapsed_time = time.time() - batch_start_time
        multi_batch_elapsed_time += batch_elapsed_time
        epoch_elapsed += batch_elapsed_time

        if step >= 0 and (step + 1) % log_step_interval == 0:
            current_datetime = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            elapsed_time = datetime.datetime.now() - start_time

            # grid = torchvision.utils.make_grid(inputs)
            # grid = util.denormalize_image(grid)
            # writer.add_image('{}/images'.format(val_name), grid, step + (len(data_loader) * (epoch - 1)),
            #                  dataformats='HWC')

            print(
                "[%s val-epoch:%d, step:%d/%d,%s] total_elapsed: %s, batch_elapsed: %f, %d steps_elapsed: %f"
                % (val_name, epoch, step, len(data_loader),
                   current_datetime, elapsed_time, batch_elapsed_time, log_step_interval,
                   multi_batch_elapsed_time))
            print("loss: %f, acc: %f, multi_batch_loss: %f, multi_batch_acc: %f" % (
                loss.item(), float(torch.sum(preds == labels.data).cpu().numpy()) / batch_size,
                multi_batch_loss / log_step_interval,
                multi_batch_corrects / (log_step_interval * batch_size)))
            multi_batch_elapsed_time = 0.
            multi_batch_loss = 0.
            multi_batch_corrects = 0

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = float(running_corrects) / len(data_loader.dataset)

    elapsed_time = datetime.datetime.now() - start_time
    current_datetime = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')

    tmp_acc = accuracy_score(epoch_labels, epoch_preds)
    f1 = f1_score(epoch_labels, epoch_preds, average='macro')
    print("[result_%s_val-epoch:%d,%s] total_elapsed: %s, epoch_elapsed: %s, loss: %f, acc: %f, f1: %f, acc2: %f" % (
        val_name, epoch, current_datetime, elapsed_time, epoch_elapsed, epoch_loss, epoch_acc, f1, tmp_acc))
    if save_confusion_matrix:
        cls_report = classification_report(epoch_labels, epoch_preds)  # , target_names=classes)
        print("{} val classification_report".format(val_name))
        print(cls_report)
        with open(os.path.join(log_dir, "eval_report_%s.txt" % val_name), "a+") as fp:
            fp.write("[epoch:%02d,%s] total_elapsed: %s, loss: %f, acc: %f\n" % (epoch, current_datetime,
                                                                                 elapsed_time, epoch_loss,
                                                                                 epoch_acc))
            epoch_cm = confusion_matrix(epoch_labels, epoch_preds)
            np_epoch_labels = np.unique(np.array(epoch_labels))
            np_epoch_labels.sort()
            log_confusion_matrix(writer, epoch, epoch_cm, np_epoch_labels)
            fp.write(str(epoch_cm) + "\n")
            print("confusion matrix")
            print(epoch_cm)
            # np.save(os.path.join(log_dir, "confusion_matrix_%s_epoch_%d.npy" % (val_name, epoch)), epoch_cm)
            epoch_cm = epoch_cm.astype('float') / epoch_cm.sum(axis=1)[:, np.newaxis]
            epoch_cm = epoch_cm.diagonal()
            fp.write(str(epoch_cm) + "\n")
            print("each accuracies")
            print(epoch_cm)
            # np.save(os.path.join(log_dir, "accuracy_each_class_%s_epoch_%d.npy" % (val_name, epoch)), epoch_cm)
            fp.write(str(cls_report) + "\n")

    writer.add_scalar('Loss/epoch', epoch_loss, epoch)
    writer.add_scalar('Acc/epoch', epoch_acc, epoch)
    writer.add_scalar('F1/epoch', f1, epoch)
    is_best = False
    if f1 > best_f1 or (f1 >= best_f1 and epoch_acc > best_acc):
        best_f1 = f1
        best_acc = epoch_acc
        best_epoch = epoch
        is_best = True

    return best_f1, best_acc, best_epoch, is_best, epoch_labels, epoch_preds, running_loss, running_corrects, len(
        data_loader.dataset)


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


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    figure.canvas.draw()
    return np.array(figure.canvas.renderer._renderer)


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def log_confusion_matrix(writer, epoch, cm, class_names=None):
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)

    writer.add_image('confusion_matrix', cm_image, epoch, dataformats='HWC')


def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    # if use_cuda:
    #     index = torch.randperm(batch_size).cuda()
    # else:
    #     index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
