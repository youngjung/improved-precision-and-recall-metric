#!/usr/bin/env python3
import os
from functools import partial
from collections import namedtuple
from glob import glob
import numpy as np
from PIL import Image
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

try:
    from tqdm import tqdm, trange
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x, desc=''):
        if len(desc) > 0:
            print(desc)
        return x

    def trange(x, desc=''):
        if len(desc) > 0:
            print(desc)
        return range(x)

import dnnlib
import tensorflow as tf
from precision_recall import DistanceBlock
from precision_recall import knn_precision_recall_features
from precision_recall import ManifoldEstimator
from utils import initialize_feature_extractor

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        # self.fnames = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.fnames = glob(os.path.join(root, '**', '*.jpg'), recursive=True) + \
            glob(os.path.join(root, '**', '*.png'), recursive=True)

        self.transform = transform

    def __getitem__(self, index):
        image_path = self.fnames[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.fnames)


def get_custom_loader(dirname, image_size=224, batch_size=50, num_workers=4, num_samples=-1):
    transform = []
    transform.append(transforms.Resize([image_size, image_size]))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(transform)

    dataset = ImageFolder(dirname, transform=transform)

    if num_samples > 0:
        dataset.fnames = dataset.fnames[:num_samples]
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)
    return data_loader


def toy():
    offset = 2
    feats_real = np.random.rand(10).reshape(-1, 1)
    feats_fake = np.random.rand(10).reshape(-1, 1) + offset
    feats_real[0] = offset
    feats_fake[0] = 1
    print('real:', feats_real)
    print('fake:', feats_fake)

    with tf.Session() as sess:
        state = knn_precision_recall_features(feats_real, feats_fake, num_gpus=1)

    precision = state['precision']
    recall = state['recall']
    print('precision:', precision)
    print('recall:', recall)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path_real', type=str, help='Path to the real images')
    parser.add_argument('path_fake', type=str, help='Path to the fake images')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size to use')
    parser.add_argument('--k', type=int, default=3, help='Batch size to use')
    parser.add_argument('--num_samples', type=int, default=5000, help='number of samples to use')
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--fname_precalc', type=str, default='', help='fname for precalculating manifold')
    args = parser.parse_args()
    num_gpus = 1

    # toy problem
    if args.toy:
        print('running toy example...')
        toy()
        exit()

    # Initialize VGG-16.
    dnnlib.tflib.init_tf()
    feature_net = initialize_feature_extractor()

    # real
    dataloader = get_custom_loader(args.path_real,
                                   batch_size=args.batch_size,
                                   num_samples=args.num_samples)
    num_images = len(dataloader.dataset)
    desc = 'found %d images in ' % num_images + args.path_real
    ref_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32)
    for i, batch in tqdm(enumerate(dataloader), desc='extract features from real...', total=len(dataloader)):
        begin = i * args.batch_size
        end = min(begin + args.batch_size, num_images)
        batch = batch.numpy()
        ref_features[begin:end] = feature_net.run(batch, num_gpus=1, assume_frozen=True)

    # fake
    dataloader = get_custom_loader(args.path_fake,
                                   batch_size=args.batch_size,
                                   num_samples=args.num_samples)
    num_images = len(dataloader.dataset)
    desc = 'found %d images in ' % num_images + args.path_fake
    eval_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32)
    for i, batch in tqdm(enumerate(dataloader), desc='extract features from fake...', total=len(dataloader)):
        begin = i * args.batch_size
        end = min(begin + args.batch_size, num_images)
        batch = batch.numpy()
        eval_features[begin:end] = feature_net.run(batch, num_gpus=num_gpus, assume_frozen=True)

    # Calculate k-NN precision and recall.
    state = knn_precision_recall_features(ref_features, eval_features, num_gpus=num_gpus)
