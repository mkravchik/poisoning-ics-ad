import tensorflow as tf
import numpy as np
from generators import *


conf = dict(
    activate_last=True,
    activation=tf.nn.tanh,
    adv_it_count=40,
    adv_lr=0.2,
    att_len=1,
    att_magn=0.4,
    att_point="CUSTOM_FIXED",
    batches=10,
    code_ratio=1.5,
    find_poison=False,
    generator=SinGenerator,
    inflate_factor=2,
    it_count=10,
    layers=1,  # increasing it did not show any significant changes
    lr=0.6,
    max_adv_iter=300,
    max_clean_poison=100,
    naive=True,
    optimizer=tf.train.GradientDescentOptimizer(0.6),
    partial_attacked=[0],
    periods=5,
    randomize=True,
    retrain=False,
    retrain_points=10,
    seq_len=2,
    signal_func=np.sin,  # signal.square,#signal.sawtooth#np.sin
    silent=False,
    single_sequence=False,
    threshold=0.2,
    total_len=50,
    train_points=5,
    window=10,)

