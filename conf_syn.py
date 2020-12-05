import tensorflow as tf
import numpy as np
from generators import *

conf = dict(
    activate_last=True,
    activation=tf.nn.tanh,
    adv_it_count=40,
    adv_lr=0.2,
    att_len=40,
    att_magn=0.5,
    att_point="CUSTOM_FIXED",  # "SIN_TOP",  # "SIN_BOTTOM",# "SIN_SIDE"
    att_start=374,  # 605 #TOP
    batches=1,
    code_ratio=2,
    find_poison=False,
    generator=SinGenerator(),
    inflate_factor=2,
    it_count=30,
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
    seq_len=22,
    signal_func=np.sin,  # signal.square,#signal.sawtooth#np.sin
    silent=False,
    single_sequence=False,
    threshold=0.2,
    total_len=100,
    train_points=10,
    window=1,
)

