# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train-data', metavar='HDF5', #required=True, # temporary default values for easier testing
                    default="data/mass_case_description_train_set_scaled_0.1_normalized_cropped.h5",
                    help='training samples (HDF5)')
parser.add_argument('--val-data', metavar='HDF5', #required=True,
                    default="data/mass_case_description_test_set_scaled_0.1_normalized_cropped.h5",
                    help='test samples (HDF5)')
parser.add_argument('--epochs', metavar='EPOCHS', type=int, default=100)
parser.add_argument('--learning-rate', metavar='LR', type=float, default=0.0001)
parser.add_argument('--batch-size-train', metavar='N', type=int, default=10,
                    help='batch size for training')
parser.add_argument('--batch-size-val', metavar='N', type=int, default=10,
                    help='batch size for validation and test')
parser.add_argument('--binary-classification', action='store_true',
                    help='use binary classification instead of 3-class classification')
parser.add_argument('--oversample', action='store_true',
                    help='use oversampling to balance classes')
parser.add_argument('--reorient-train', action='store_true',
                    help='reorient all images in the same direction')
parser.add_argument('--reorient-val', action='store_true',
                    help='reorient all images in the same direction')
parser.add_argument('--filter', metavar='N', type=str, default="",
                    help='Available filters: L, R, LEFT, RIGHT, CC, MLO.'
                    'Difference between L/R and LEFT/RIGHT is that the former is calculated,'
                    'while the latter was included in the dataset and might include errors')
parser.add_argument('--num-tokens', metavar='N', type=int, default=16,
                    help='specifies the number of tokens used by transformer')
parser.add_argument('--transform-depth', metavar='N', type=int, default=6,
                    help='specifies the number transformer layers')
parser.add_argument('--dim', metavar='N', type=int, default=128,
                    help='dim')
parser.add_argument('--mlp-dim', metavar='N', type=int, default=256,
                    help='mlp_dim')
parser.add_argument('--cross-val', metavar='N', type=int, default=0,
                    help='use cross-validation with N folds')
parser.add_argument('--num-workers', type=int, default=0,
                    help='num_workers passed to train_loader')
parser.add_argument('--tensorboard-dir', metavar='DIR',
                    help='log statistics to tensorboard')
parser.add_argument('--model', metavar='DIR',
                    help='save model to given directory')
