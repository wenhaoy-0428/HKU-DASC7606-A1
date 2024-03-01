import argparse
import collections
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, collater, Resizer, \
    AspectRatioBasedSampler, Augmenter, Normalizer
from retinanet.eval import Evaluation
from torch.utils.data import DataLoader

import hyper_param as HYPER_PARAM

from torch.utils.tensorboard import SummaryWriter
import shutil


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='./data')
    parser.add_argument('--output_path', help='Path to output directory to save checkpoints', default='./output')
    parser.add_argument('--depth', help='ResNet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=72)

    parser = parser.parse_args(args)
    parser.output_path = "./output_lr{}_d{}_e{}_b{}".format(str(HYPER_PARAM.learning_rate).replace('-', '_'), parser.depth, parser.epochs, HYPER_PARAM.batch_size)
    print(parser.output_path)
    if not os.path.exists(parser.output_path):
        os.mkdir(parser.output_path)

    shutil.copy("./hyper_param.py", parser.output_path)

    if parser.coco_path is None:
        raise ValueError('Must provide --coco_path when training on COCO.')

    dataset_train = CocoDataset(parser.coco_path, set_name='train',
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CocoDataset(parser.coco_path, set_name='val',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=HYPER_PARAM.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=2, collate_fn=collater, batch_sampler=sampler)

    # Create the model
    if parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()
        if torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            retinanet = retinanet.to(mps_device)

    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=HYPER_PARAM.learning_rate)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[48, 64])

    loss_hist = collections.deque(maxlen=500)
    epoch_loss_list = []


    writer = SummaryWriter(log_dir='./tb')
    print('Num training images: {}'.format(len(dataset_train)))
    for epoch_num in range(parser.epochs):

        retinanet.training = True
        retinanet.train()
        retinanet.freeze_bn()

        epoch_loss = []

        for iter_num, data in tqdm(enumerate(dataloader_train)):

            ###################################################################
            # TODO: Please fill the codes here to zero optimizer gradients
            ##################################################################
            optimizer.zero_grad()

            ##################################################################

            if torch.cuda.is_available():
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda()])
            elif torch.backends.mps.is_available():
                mps_device = torch.device("mps")
                classification_loss, regression_loss = retinanet(
                    [data['img'].to(mps_device).float(), data['annot'].to(mps_device)])
            else:
                classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            ###################################################################
            # TODO: Please fill the codes here to complete the gradient backward
            ##################################################################
            loss.backward()

            ##################################################################

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            ###################################################################
            # TODO: Please fill the codes here to optimize parameters
            ##################################################################
            optimizer.step()

            ##################################################################

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            if iter_num % 100 == 0:
                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
            

            writer.add_scalar( f'Epoch {epoch_num} Classification loss/train',float(classification_loss),iter_num)
            writer.add_scalar( f'Epoch {epoch_num} Regression loss/train',float(regression_loss),iter_num)
            writer.add_scalar( f'Epoch {epoch_num} Running loss/train',np.mean(loss_hist),iter_num)

            del classification_loss
            del regression_loss

        scheduler.step()
        writer.add_scalar('Running loss per Epoch/train',np.mean(epoch_loss),epoch_num)

        epoch_loss_list.append(np.mean(epoch_loss))

        if (epoch_num + 1) % HYPER_PARAM.save_per_epoch == 0 or epoch_num + 1 == parser.epochs:
            print('Evaluating dataset')
            retinanet.eval()
            retinanet.training = False
            eval = Evaluation()
            eval.evaluate(dataset_val, retinanet)

            torch.save(retinanet, os.path.join(parser.output_path, 'retinanet_epoch{}.pt'.format(epoch_num + 1)))

    print(epoch_loss_list)
    torch.save(retinanet, os.path.join(parser.output_path, 'model_final.pt'))
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
