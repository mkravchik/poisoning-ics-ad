import sys
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time, datetime
import tensorflow as tf
import collections
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error
from math import sqrt
from munch import *
from tqdm import tqdm
import os
import glob
import copy, random
import seaborn as sns
import matplotlib as mpl
import argparse
import importlib
from generators import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
data_path = "./"


def model_signal(conf, model_name, src, restore=False):
    random.seed(1)
    c = munchify(conf)
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
        if restore:
            saver = tf.train.import_meta_graph(model_name + ".meta")
        else:
            # once the new graph is set to be default, we can fix its seed
            tf.set_random_seed(1)
            # Using float64 instead of the default float32 gives much faster run and slightly better results
            x = tf.placeholder("float64", [None, c.seq_len, src.shape[-1]], name="x")
            flat = tf.reshape(x, (-1, x.shape[-2] * x.shape[-1]), name="flat_x")

            if c.inflate_factor:
                inflate_size = x.shape[-2] * x.shape[-1] * c.inflate_factor
            else:
                inflate_size = x.shape[-2] * x.shape[-1]
            code_size = max(((x.shape[-2] * x.shape[-1]).value) // c.code_ratio, 1)
            sizes = np.linspace(inflate_size.value, code_size, c.layers + 1).astype(int)
            print(sizes)

            if c.inflate_factor:
                enc_l = tf.layers.dense(flat, inflate_size, tf.nn.tanh, name="encoder_inflator")
            else:
                enc_l = flat

            for l_i, layer_size in enumerate(sizes[1:]):
                enc_l = tf.layers.dense(enc_l, layer_size, tf.nn.tanh, name="encoder_%d" % l_i)
                if not c.silent: print("Encoder: Adding dense ", layer_size)
            dec_l = enc_l
            if c.inflate_factor:
                dec_l = tf.layers.dense(dec_l, inflate_size, tf.nn.tanh, name="decoder_inflator")
            for l_i, layer_size in enumerate(sizes[:-1][::-1]):
                dec_l = tf.layers.dense(dec_l, layer_size, tf.nn.tanh, name="decoder_%d" % l_i)
                if not c.silent: print("Decoder: Adding dense ", layer_size)

            # it is important to NOT have any activation in the last layer
            # as the activations limit the output to their range
            dec_l = tf.layers.dense(dec_l, (x.shape[-2] * x.shape[-1]), name="decoder_last")
            if c.activate_last:
                dec_l = tf.nn.tanh(dec_l, name="decoder_last_activation")

            out = tf.reshape(dec_l, [-1, c.seq_len, src.shape[-1]], name="out")
            diff = tf.subtract(x, out, name="diff")
            error = tf.reduce_mean(tf.square(diff), name="adv_error")
            optimizer = c.optimizer
            train = optimizer.minimize(error, name='train')

            init = tf.global_variables_initializer()

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config, graph=g) as sess:
        if len(src.shape) == 2:
            batches = c.batches
            #             #round the length so that it is divisible by batch_size
            rounded_len = len(src) // batches * batches
            src = src[:rounded_len, :]
            org_shape = src.shape
            src = src.reshape((batches, rounded_len // batches, -1))
        else:
            # src will arrive as an array [train_points, steps, features]
            # and we will train on all of the samples simultaneously
            org_shape = src.shape
            pass

        if restore:
            saver.restore(sess, model_name)
        else:
            # Run the initializer
            sess.run(init)

        # Show what we have in the graph
        trainable_vars = tf.trainable_variables()
        weights = []
        for v in trainable_vars:
            if v.name.find('kernel') != -1:
                if not c.silent: print("Layer", v.name, v)
                weights.append(v)

        loop = tqdm(range(c.it_count))
        for it in loop:
            err_acc = 0
            # run over all samples
            if "single_sequence" in c and c.single_sequence:
                err_acc, _ = sess.run(["adv_error:0", "train"], feed_dict={"x:0": src})
            else:
                # split into batches for speedup
                num_seqs = src.shape[-2] - c.seq_len + 1
                # randomize
                # IMPORTANT - without randomization the last training
                # batch is pulling the weights in its
                # direction thus producing inferior prediction
                # as all my signals are aligned in form
                batch_indices = np.arange(num_seqs)
                if c.randomize:
                    random.shuffle(batch_indices)
                for b in batch_indices:
                    res = sess.run(["adv_error:0", "out:0", "train"], feed_dict={"x:0": src[:, b:b + c.seq_len, :]})
                    err_acc += res[0] / num_seqs
            if not c.silent:
                if it % 10 == 0 or it == c.it_count - 1: print("Epoch %d, avg. error %f" % (it, err_acc))

        # now run the test
        err_acc = 0
        # run over all samples
        if "single_sequence" in c and c.single_sequence:
            err_acc, n_out = sess.run(["adv_error:0", "out:0"], feed_dict={"x:0": src})
        else:
            net_out = []
            num_seqs = src.shape[-2] - c.seq_len + 1

            for b in range(num_seqs):
                res = sess.run(["adv_error:0", "out:0"], feed_dict={"x:0": src[:, b:b + c.seq_len, :]})
                #                 print(res[0])
                err_acc += res[0] / num_seqs
                if b == 0:
                    net_out.append(res[1])
                else:
                    # take the last element only
                    net_out.append(res[1][:, -1:, :])

            n_out = np.concatenate(net_out, axis=1)
        if not c.silent:
            n_out = n_out.reshape(org_shape)
            src = src.reshape(org_shape)
            if len(org_shape) == 2:
                src = src.reshape((1, -1, org_shape[-1]))
                n_out = n_out.reshape((1, -1, org_shape[-1]))

            roll_errors = []
            for sig_i in [0]:  # range(len(src)):
                diffs = np.abs(src[sig_i, :, :] - n_out[sig_i, :, :])
                for col in range(src.shape[-1]):
                    roll_diffs = pd.Series(diffs[:, col]).rolling(c.window, min_periods=1).min()
                    col_alerts = (roll_diffs > c.threshold).sum()
                    roll_errors.append(roll_diffs.max())
                    if col_alerts:
                        print("Signal %d still raises %d alerts" % (sig_i, col_alerts))

            print("Max model error", np.max(np.abs(src - n_out)), "max rolling error", np.max(roll_errors))
        saver = tf.train.Saver()
        saver.save(sess, model_name)
        return err_acc


def poison_desc(poison, reference):
    return sqrt(mean_squared_error(poison.reshape((-1, poison.shape[-1])),
                                   reference.reshape((-1, poison.shape[-1]))))


def test_input(sess, x_i, c, display=False, title=""):
    mpl.rcParams.update({'font.size': 16})
    if "single_sequence" in c and c.single_sequence:
        n_out, test_err = sess.run(["out:0", "adv_error:0"], feed_dict={"x:0": x_i})
    else:
        x_i = x_i.copy().reshape((1, -1, x_i.shape[-1]))
        net_out = []
        err_acc = 0
        for b in range(x_i.shape[-2] - c.seq_len + 1):
            res = sess.run(["adv_error:0", "out:0"], feed_dict={"x:0": x_i[:, b:b + c.seq_len, :]})
            err_acc += res[0]
            if b == 0:
                net_out.append(res[1])
            else:
                # take the last element only
                net_out.append(res[1][:, -1:, :])

        test_err = err_acc / (x_i.shape[-2] - c.seq_len + 1)
        n_out = np.concatenate(net_out, axis=1)

    n_out_flat = n_out.reshape((-1, n_out.shape[-1]))
    x_i_flat = x_i.reshape((-1, x_i.shape[-1]))
    diffs = np.abs(x_i_flat - n_out_flat)
    alerts = 0
    roll_errors = []
    for col in range(n_out.shape[-1]):
        roll_diffs = pd.Series(diffs[:, col]).rolling(c.window, min_periods=1).min()
        col_alerts = (roll_diffs > c.threshold).sum()
        roll_errors.append(roll_diffs.max())
        alerts += col_alerts

    if display:
        for col in range(n_out.shape[-1]):
            plt.figure()
            plt.plot(x_i_flat[:, col], label="Attack Input")
            plt.plot(n_out_flat[:, col], label="Poisoned Attack Output")
            plt.plot(np.abs(x_i_flat[:, col] - n_out_flat[:, col]), label="Residue")
            plt.legend()
            plt.ylabel("Signal")
            plt.xlabel("Timepoint")
            plt.yticks(np.arange(-1.6, 1.6, step=0.2))
            plt.title(title)

    return test_err, n_out, alerts, np.max(roll_errors)


def build_attack_signal(input_sig, c):
    poisoned_point = 0
    x_i_att = copy.copy(input_sig[0]) + c.att_magn
    x_poison = copy.copy(input_sig[0])
    for i in range(2, len(input_sig)):
        if c.att_point == "BOTTOM":
            cond = (input_sig[i][0] == -1 and input_sig[i - 1][0] == -1)
        elif c.att_point == "TOP":
            cond = (input_sig[i][0] == 1 and input_sig[i - 1][0] == 1)
        elif c.att_point == "SIDE":
            cond = (input_sig[i][0] == -1)
        elif c.att_point == "SIN_BOTTOM":
            cond = (np.abs(input_sig[i][0] + 0.5) <= 0.01)
        elif c.att_point == "SIN_TOP":
            cond = (np.abs(input_sig[i][0] - 0.5) <= 0.01)
        elif c.att_point == "SIN_SIDE":
            cond = (np.abs(input_sig[i][0]) <= 0.1)
        elif c.att_point == "CUSTOM" or c.att_point == "CUSTOM_FIXED" or c.att_point == "CUSTOM_LINEAR":
            cond = (i == c.att_start)
        else:
            cond = True
        if cond:
            partial = None if not "partial_attacked" in c else c["partial_attacked"]
            x_poison = copy.copy(input_sig[i:i + c.att_len, :])
            x_i_att = copy.copy(input_sig[i:i + c.att_len, :])

            def update_sig(sig):
                if c.att_point == "CUSTOM_FIXED":
                    sig = c.att_magn
                elif c.att_point == "CUSTOM_LINEAR":
                    addition = np.linspace(0, c.att_magn, c.att_len).reshape((-1, 1))
                    sig += addition
                else:
                    sig += c.att_magn
                return sig

            x_i_att = c.generator.apply(x_i_att, update_sig, partial)
            poisoned_point = i
            break

    return x_i_att, x_poison, poisoned_point


def generate_poison_and_attack(conf):
    c = munchify(conf)
    input_sig, t = c.generator(c)

    input_len = len(input_sig)
    x_i_att, x_poison, poisoned_point = build_attack_signal(input_sig, c)

    x_att_full = copy.copy(input_sig)
    x_att_full[poisoned_point:poisoned_point + c.att_len, :] = x_i_att

    return input_sig.copy(), x_att_full


def train_and_test_adv(model_name, model_idx, it_count, x_train, complete_poison,
                       x_poison, x_test, c, train_curr_poison=True,
                       display=False, title=""):
    if c.randomize:
        random.seed()
    else:
        random.seed(1)

    def train_and_test_adv_internal(model_name, model_idx, it_count, x_train, complete_poison,
                                    x_poison, x_test, c, train_curr_poison=True,
                                    display=False, title=""):
        if x_train is not None:
            if len(x_train.shape) == 3 and x_train.shape[0] != c.batches:
                #       if not c.silent: print("Reshaping from ", x_train.shape,
                #                               "into %d batches"% c.batches)
                # will be reshaped by the next line
                x_train = x_train.reshape((-1, x_train.shape[-1]))

            if len(x_train.shape) == 2:
                # With real signals I have training_points * att_len * features, so if I reshape it to
                # batches * att_len * features it can't be concatenated with the poison, which is 1 * att_len * features
                att_len = x_poison.shape[-2]
                rounded_len = len(x_train) // att_len * att_len
                x_train = x_train[:rounded_len, :]
                x_train = x_train.reshape((-1, att_len, x_train.shape[-1]))

        with tf.Graph().as_default():
            tf.set_random_seed(1)
            config = tf.ConfigProto(log_device_placement=False)
            config.gpu_options.allow_growth = True
            train_err = 0
            with tf.Session(config=config) as sess:
                if it_count:  # we have some training to do
                    tf.train.import_meta_graph(model_name + ".meta").restore(sess, model_name)

                    # build combined input
                    inp_srcs = [x_train]
                    if len(complete_poison):
                        inp_srcs.append(np.array([p[0, :, :] for p in complete_poison]))
                    if train_curr_poison:
                        inp_srcs.append(x_poison)

                    # there is problem with the form of batches
                    inp = np.concatenate(inp_srcs)
                    #                 if not c.silent: print("inp", inp.shape, "it_count", it_count)

                    for it in range(it_count):
                        if "single_sequence" in c and c.single_sequence:
                            train_err, _ = sess.run(["adv_error:0", "train"], feed_dict={"x:0": inp})
                        else:
                            train_err = 0
                            err_acc = 0
                            num_seqs = x_train.shape[-2] - c.seq_len + 1
                            batch_indices = np.arange(num_seqs)
                            if c.randomize:
                                random.shuffle(batch_indices)
                            for b in batch_indices:
                                err, _ = sess.run(["adv_error:0", "train"],
                                                  feed_dict={"x:0": inp[:, b:b + c.seq_len, :]})
                                err_acc += err
                            train_err = err_acc / (x_train.shape[-2] - c.seq_len + 1)
                else:  # no training, restore the weights
                    tf.train.import_meta_graph(model_name + ".meta").restore(sess, model_name)

                # Run on the entire input
                test_err, n_out, alerts, max_err = test_input(sess, x_test, c, display, title)

                x_val, _ = c.generator(c)
                x_val = x_val.reshape((1, x_test.shape[-2], x_test.shape[-1]))
                val_err, _, val_alerts, _ = test_input(sess, x_val, c, display, title + " for validation ")

                # We are going to save and restore later to be wrapped
                model_file_name = model_name + "_%d" % model_idx
                tf.train.Saver().save(sess, model_file_name)

        return train_err, test_err, n_out, alerts, model_file_name, max_err, val_alerts

    if c.get("sec_seq_len") is None:
        return train_and_test_adv_internal(model_name, model_idx, it_count, x_train, complete_poison,
                                           x_poison, x_test, c, train_curr_poison, display, title)
    else:
        #         if not c.silent: print("Running with seq_len", c.seq_len)
        res1 = train_and_test_adv_internal(model_name, model_idx, it_count, x_train, complete_poison,
                                           x_poison, x_test, c, train_curr_poison, display, title)

        #         if not c.silent: print("Running with sec_seq_len", c.sec_seq_len)
        tmp = c.seq_len
        c.seq_len = c.sec_seq_len
        res2 = train_and_test_adv_internal(c.sec_model_name, model_idx, it_count, x_train, complete_poison,
                                           x_poison, x_test, c, train_curr_poison, display, title)
        c.seq_len = tmp
        # return the biggest error
        if res1[5] > res2[5]:
            #             if not c.silent: print ("returning", res1)
            return res1
        else:
            #             if not c.silent: print ("returning", res2)
            return res2


# The comments in the code refer to the back gradient optimization implementation at:
# https://github.com/lmunoz-gonzalez/Poisoning-Attacks-with-Back-gradient-Optimization
def back_gradient_optimization(sess, x_poison, it_count, epsilon, lr, adv_lr,
                               t_grad_x, weights, t_weights_grads,
                               weights_vals, weights_grads_vals, mask=None):
    if mask is None:
        mask = np.ones(x_poison.shape)
    # Rollback the iterations
    dxp = np.zeros(x_poison.shape)

    for it in range(it_count):
        # %% Line 4 g_(t-1) <= grad(L(x_c, w_t))
        # Get weights' gradients on the poison input
        # [c,dww] = getDerivativesMLP(x2,y2,net);
        t_weights_grads_ = sess.run(t_weights_grads, feed_dict={"x:0": x_poison})
        t_weights_poison_grads = np.array([a[0] for a in t_weights_grads_])

        # Use a trick to calculate the hessian-vector product without calculating it
        # The trick
        # wwm = wwm + 0.5.*epsilon.*dw;
        for w_i, w in enumerate(weights):
            sess.run(tf.assign(w, weights_vals[w_i] + 0.5 * epsilon * weights_grads_vals[w_i]))

        # %% Derivative of error relative to x
        # [c2x,dw2x] = getDerivativesMLP2(x2,y2,net2);
        # %% Derivative of error relative to w
        # [c2,dw2] = getDerivativesMLP(x2,y2,net2);
        [g_x_p] = sess.run([t_grad_x], feed_dict={"x:0": x_poison})
        g_x_p = g_x_p[0]
        t_weights_grads_ = sess.run(t_weights_grads, feed_dict={"x:0": x_poison})
        t_weights_poison_grads_plus = np.array([a[0] for a in t_weights_grads_])

        # wwm = wwm - epsilon.*dw;
        for w_i, w in enumerate(weights):
            sess.run(tf.assign(w, weights_vals[w_i] - 0.5 * epsilon * weights_grads_vals[w_i]))

        # %% Derivative of error relative to x
        # [c2x,dw2x] = getDerivativesMLP2(x2,y2,net2);
        # %% Derivative of error relative to w
        # [c2,dw2] = getDerivativesMLP(x2,y2,net2);
        [g_x_m] = sess.run([t_grad_x], feed_dict={"x:0": x_poison})
        g_x_m = g_x_m[0]
        t_weights_grads_ = sess.run(t_weights_grads, feed_dict={"x:0": x_poison})
        t_weights_poison_grads_minus = np.array([a[0] for a in t_weights_grads_])

        # ddxp = (dw2x - dw1x)./epsilon;
        ddxp = (g_x_p - g_x_m) / epsilon

        # ddw = (dw2 - dw1)./epsilon;
        ddw = (t_weights_poison_grads_plus - t_weights_poison_grads_minus) / epsilon
        # End of Trick

        # %% Line 2 dx_c = dx_c - alpha * ...
        # dxp = dxp - alpha.*ddxp'; %'
        # NOTE - we rely upon knowning the LR!!!
        dxp = dxp - lr * ddxp

        # %% Line 3 dw = dw - alpha * ...
        # dw = dw - alpha.*ddw;
        weights_grads_vals -= lr * ddw

        # %% Line 5 w_(t-1) = w_t + alpha*g_(t-1)
        # For the next iteration
        # ww = ww + alpha.*dww;
        weights_vals += lr * t_weights_poison_grads
        for w_i, w in enumerate(weights):
            sess.run(tf.assign(w, weights_vals[w_i]))

            # This is present in the algorithm and absent from the code?!
    # Add the gradient of the input
    # dxp += grad_x[0]

    # I want to minimize the error thus -
    # Calculate the max of the absolute gradient values.
    # This is used to calculate the step-size.
    grad_absmax = np.abs(dxp).max()

    # If the gradient is very small then use a lower limit,
    # because we will use it as a divisor.
    if grad_absmax < 1e-10:
        grad_absmax = 1e-10

    step_size = adv_lr / grad_absmax

    # apply partial mask
    dxp *= mask

    x_poison -= step_size * dxp
    return x_poison


def prepare_graph_for_optimization(sess, model_file_name, c, silent=True, input_len=1, xdims=1):
    if "single_sequence" in c and c.single_sequence:
        tf.train.import_meta_graph(model_file_name + ".meta").restore(sess, model_file_name)

        x = tf.get_default_graph().get_operation_by_name('x').outputs[-1]

        error = tf.get_default_graph().get_operation_by_name('adv_error').outputs[-1]

    else:  # WRAPPING
        x = tf.placeholder("float64", [1, input_len, xdims], name="x")
        x_exp = tf.expand_dims(x, 0)
        x_patches = tf.extract_image_patches(images=x_exp, ksizes=[1, 1, c.seq_len, 1],
                                             strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        x_patches = tf.squeeze(x_patches)
        x_patches = tf.reshape(x_patches, (-1, c.seq_len, xdims))

        # restore and bind
        tf.train.import_meta_graph(model_file_name + ".meta", input_map={"x:0": x_patches}).restore(sess,
                                                                                                    model_file_name)
        out_patches = tf.get_default_graph().get_operation_by_name("out").outputs[-1]
        out_reordered = out_patches[:, -1, :]

        out_restored = tf.reshape(out_reordered, (1, -1, xdims))

        diff = tf.subtract(
            x[:, -out_restored.shape[-2].value:, :],
            out_restored[:, :, :])

        # It is not a good idea to use clipped error, as there is no gradients on poison
        # and thus poison is not updated
        # error = tf.reduce_mean(tf.square(diff_clip), name = "full_error_clip")
        # Using reduce_sum instead of reduce_mean provides larger gradients
        error = tf.reduce_mean(tf.square(diff)) + tf.reduce_max(tf.abs(diff))

    # add gradients to the graph
    t_grad_x = tf.gradients(error, x)
    t_weights_grads = []
    trainable_vars = tf.trainable_variables()

    weights = []
    for v in trainable_vars:
        if v.name.find('kernel') != -1:
            if not silent: print("Adding", v.name, v)
            weights.append(v)

    for w in weights:
        t_weights_grads.append(tf.gradients(error, w))

    return t_grad_x, weights, t_weights_grads, error


def poison_model(conf, model_name, start_poison, x_att, x_train):
    # remove old intermediate models
    for f in glob.glob("." + os.sep + "tmp" + os.sep + model_name + '_*.*'):
        os.remove(f)

    c = munchify(conf)
    xdims = start_poison.shape[-1]
    input_len = len(start_poison)

    x_poison = start_poison.copy().reshape((1, -1, xdims))
    x_i_att = x_att.copy().reshape((1, -1, xdims))
    if len(x_train.shape) == 2:
        x_train = x_train.reshape((1, -1, xdims))

    # Testing incomplete training
    it_count = c.adv_it_count
    failed_updates = 0
    orig_poison = x_poison.copy()

    complete_poison = []

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    adv_lr = c.adv_lr
    loop = tqdm(range(c.max_adv_iter))
    poisons = []
    model_file_name = model_name

    train_err = 0
    test_err = 0
    alerts = 0
    max_err = 0
    val_alerts = 0
    i = 0
    # partial mask
    mask = None
    if "partial_attacked" in c:
        mask = np.zeros_like(x_i_att)
        for col in c["partial_attacked"]:
            mask[:, :, col] = 1

    for i in loop:
        poisons.append(x_poison)

        # WRAPPING
        # Re-Define a placeholder that will receive a single-batch long input
        tf.reset_default_graph()
        tf.set_random_seed(1)
        with tf.Session(config=config) as sess:
            t_grad_x, weights, t_weights_grads, error = prepare_graph_for_optimization(sess, model_file_name, c,
                                                                                       i != 0 or c.silent, input_len,
                                                                                       xdims)

            # For 32 bits I can use only 0.0001, smaller values result in 0 second gradients
            # However, I haven't noticed any improvement in results with smaller epsilons
            epsilon = 1e-8
            fetch = [t_grad_x, error]
            weights_idx = len(fetch)
            fetch.extend(weights)
            weights_grad_idx = len(fetch)
            fetch.extend(t_weights_grads)

            # Get the final gradients, weights and error on the (full) attack input
            # should I assess it on the attack + validation set?!
            res = sess.run(fetch, feed_dict={"x:0": x_i_att})

            full_error = res[1]
            weights_vals = np.array(res[weights_idx:weights_grad_idx])
            weights_grads_vals = np.array([a[0] for a in res[weights_grad_idx:]])

            poison_dist = poison_desc(orig_poison, x_poison)

            adv_lr_adjusted = False
            prev_x_poison = x_poison.copy()
            x_poison = back_gradient_optimization(sess, x_poison, it_count, epsilon, c.lr, adv_lr,
                                                  t_grad_x, weights, t_weights_grads,
                                                  weights_vals, weights_grads_vals, mask)

        # let's check if this poison will cause an alert!
        _, p_err, _, alerts, _, _, val_alerts = train_and_test_adv(model_name, i, c.it_count, x_train,
                                                                   complete_poison, x_poison, x_poison, c,
                                                                   train_curr_poison=False,
                                                                   display=False,
                                                                   title="Iteration %d. New poison: " % i)
        if alerts:
            if not c.silent: print("The new poison (%f) will cause " % poison_desc(orig_poison, x_poison),
                                   alerts, "alerts and %d val_alerts" % val_alerts, "Poison error", p_err,
                                   "previous poison",
                                   poison_desc(orig_poison, prev_x_poison))
            if np.max(np.abs(prev_x_poison - orig_poison)):  # don't add zero poison
                if not c.silent:
                    print("Adding another poison point.", poison_desc(orig_poison, prev_x_poison))
                complete_poison.append(prev_x_poison)

            # if the poison creates alerts, its' too aggressive - decrease the learning rate
            failed_updates += 1
            # speed up the step decreasing if we keep on failing
            adv_lr *= (0.90 ** failed_updates)
            if adv_lr <= 0.00001:
                if not c.silent: print("The learning rate (%f) is already small, exiting!"
                                       % (adv_lr))
                break
            # let's try with the last successful poison
            x_poison = prev_x_poison.copy()
            continue

        if val_alerts:
            if not c.silent: print("The new poison (%f) will cause " % poison_desc(orig_poison, x_poison),
                                   "%d val_alerts" % val_alerts, "Poison error", p_err, "previous poison",
                                   poison_desc(orig_poison, prev_x_poison))
            while val_alerts and len(complete_poison) < 200:
                complete_poison.append(x_train[-1:, -start_poison.shape[-2]:, :])
                if not c.silent: print("Adding clean data.")
                _, p_err, _, alerts, _, _, val_alerts = train_and_test_adv(model_name, i, c.it_count, x_train,
                                                                           complete_poison, x_poison, x_poison, c,
                                                                           train_curr_poison=False,
                                                                           display=False)
            # keep this poison and move to the testing
            if len(complete_poison) >= 200:
                if not c.silent: print("Too many poison points %d, exiting!" % (complete_poison))
                break

        elif not adv_lr_adjusted:  # no alerts on poison? restore the learning rate, let's try to be aggressive
            adv_lr /= 0.98
            adv_lr = np.clip(adv_lr, 0, 1)
            failed_updates = 0

        train_err, test_err, _, alerts, model_file_name, \
        max_err, val_alerts = train_and_test_adv(model_name, i, c.it_count, x_train, complete_poison, x_poison, x_i_att,
                                                 c)
        if not c.silent:
            print("Iteration %d, poison %f, training error %f, test error %f, %d alerts full err %f adv_lr "
                  "%f, max err %f, val_alerts %d" %
                  (i, poison_dist, train_err, test_err, alerts, full_error, adv_lr, max_err, val_alerts))
            print("complete_poison", [poison_desc(orig_poison, p) for p in complete_poison])
        if not alerts + val_alerts:
            if not c.silent: print("We are done!!!")
            break
        elif val_alerts:
            while val_alerts and len(complete_poison) < 200:
                complete_poison.append(x_train[-1:, -start_poison.shape[-2]:, :])  # The amount of added points can be
                # adjustable
                if not c.silent: print("Adding clean data.")
                train_err, test_err, _, alerts, model_file_name, max_err, val_alerts = train_and_test_adv(model_name, i,
                                                                                                          c.it_count,
                                                                                                          x_train,
                                                                                                          complete_poison,
                                                                                                          x_poison,
                                                                                                          x_i_att, c)
                if not c.silent: print("%d alerts adv_lr %f max err %f val_alerts %d" %
                                       (alerts, adv_lr, max_err, val_alerts))
                if not alerts + val_alerts:
                    if not c.silent: print("We are done!!!")
                    break

    # Final result
    if not c.silent: print("Got ", len(complete_poison) + 1, " poison points. Doing final check")

    print("Final poisoned training error %f, test error %f, %d alerts, %d val alerts, max_err %f" % (
        train_err, test_err, alerts, max_err, val_alerts))

    return alerts + val_alerts, len(complete_poison) + 1, i, model_file_name


def poison_model_naive(conf, model_name, start_poison, x_att, x_train):
    """
      param: start_poison - (length, features)
      param: x_att - (length, features)
    """
    c = munchify(conf)
    xdims = start_poison.shape[-1]
    x_poison = start_poison.copy().reshape((1, -1, xdims))
    x_i_att = x_att.copy().reshape((1, -1, xdims))
    if len(x_train.shape) == 2:
        x_train = x_train.reshape((1, -1, xdims))

    # remove old intermediate models
    for f in glob.glob("." + os.sep + "tmp" + os.sep + model_name + '_*.*'):
        os.remove(f)

    #     if not c.silent:
    #         #show the unpoisoned weights
    #         org_weights = show_model_weights(model_name)

    curr_poison = x_poison.copy()
    poisons = []
    rate = 1
    step_size = 1
    i = 0
    prev_poison = curr_poison.copy()
    loop = tqdm(range(c.max_adv_iter))
    it_count = c.it_count
    smallest_step = 0.001

    # How does the model look like without any poisoning?
    train_err, test_err, n_out, alerts, \
    _, max_error, val_alerts = train_and_test_adv(model_name, i, it_count, x_train,
                                                  poisons,
                                                  curr_poison, x_i_att, c,
                                                  train_curr_poison=False,
                                                  display=False,
                                                  title="Test with NO poison")
    if not c.silent: print(
            "No-poison Test: tr. error %f, test error %f rate %f, %d alerts max err %f %d val alerts %f" % (
        train_err, test_err, rate, alerts, max_error, val_alerts, np.max(np.abs(step_size))))

    model_file_name = ""
    for i in loop:
        if np.max(np.abs(step_size)) >= smallest_step or 0 == len(
                poisons):  # if we have no poisons found, disregard the lower limit
            step_size = rate * (x_i_att - prev_poison) / 2
            curr_poison = prev_poison + step_size

            # test whether the new poison raises alert
            train_err, test_err, _, \
            alerts, _, max_error, val_alerts = train_and_test_adv(model_name, i, it_count,
                                                                  x_train, poisons,
                                                                  curr_poison, curr_poison, c,
                                                                  train_curr_poison=False,
                                                                  display=False,
                                                                  title="Current poison test ")
            if not c.silent: print(
                    "Poison %f tr. error %f, test error %f rate %f, %d alerts max err %f %d val alerts %f" % (
                poison_desc(x_poison, curr_poison),
                train_err, test_err, rate, alerts, max_error, val_alerts, np.max(np.abs(step_size))))
            if alerts:
                # decrease the step
                rate *= 0.9
                continue
        else:
            # just add the last working poison
            curr_poison = prev_poison
            # increase the step - we are on the border of the smallest addition
            rate /= 0.9
            step_size = np.max([np.max(np.abs(step_size)),
                                smallest_step])  # so we try the next time, the sign does not matter, as the step is
            # recalculated each time
            if not c.silent: print("Trying last working poison %f " % poison_desc(x_poison, curr_poison))
        # the poison does not cause alerts, add it
        step_size = np.max([np.max(np.abs(step_size)), smallest_step])  # make sure we will continue the next iteration
        poisons.append(curr_poison)
        # test
        train_err, test_err, n_out, alerts, \
        model_file_name, max_error, val_alerts = train_and_test_adv(model_name, i,
                                                                    it_count,
                                                                    x_train,
                                                                    poisons,
                                                                    curr_poison,
                                                                    x_i_att, c,
                                                                    train_curr_poison=False,
                                                                    display=False,
                                                                    title="Test with poison %s " %
                                                                          poison_desc(x_poison, curr_poison))
        if not c.silent: print(
                "Test: Poison %f tr. error %f, test error %f rate %f, %d alerts max err %f %d val alerts %f" % (
            poison_desc(x_poison, curr_poison),
            train_err, test_err, rate, alerts, max_error, val_alerts, np.max(np.abs(step_size))))
        if alerts == 0:
            if not c.silent: print("We are done! Got %d poison points and no alerts" % len(poisons))
            break
        elif val_alerts:
            if not c.silent: print("Got %d val_alerts with poison. Adding clean data." % val_alerts,
                                   poison_desc(x_poison, curr_poison))
            # decrease the step
            rate *= 0.9
            del poisons[-1]  # remove the tried poison
            poisons.append(x_train[-1:, -start_poison.shape[-2]:, :])  # The amount of added points can be adjustable
            continue  # don't change the prev_poison
        else:
            # increase the step
            rate /= 0.9

        # add the poison
        if not c.silent: print("Added poison ", poison_desc(x_poison, curr_poison))
        prev_poison = curr_poison.copy()

        i += 1
    return alerts + val_alerts, len(poisons), i + 1, model_file_name


def find_start_poison(conf, model_name, x_att, x_train):
    c = munchify(conf)
    # remove old intermediate models
    for f in glob.glob("." + os.sep + "tmp" + os.sep + model_name + '_*.*'):
        os.remove(f)

    c = munchify(conf)
    xdims = x_att.shape[-1]
    input_len = len(x_att)
    x_poison = x_att.copy().reshape((1, -1, xdims))

    all_markers = list(Line2D.markers.keys())

    prev_adv_errors = collections.deque(2 * [1000000000], 2)
    failed_updates = 0
    orig_poison = x_poison.copy()
    prev_x_poison = x_poison.copy()

    complete_poison = []

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    # Initial test
    train_err, test_err, n_out, alerts, _, max_err, _ = train_and_test_adv(model_name, 0, 0, x_train,
                                                                           [], x_poison,
                                                                           x_poison, c, False, not c.silent,
                                                                           title="Initial test")

    adv_lr = c.adv_lr
    orig_adv_lr = adv_lr
    # WRAPPING
    # Re-Define a placeholder that will receive a single-batch long input
    tf.reset_default_graph()
    tf.set_random_seed(1)
    with tf.Session(config=config) as sess:
        if "single_sequence" in c and c.single_sequence:
            tf.train.import_meta_graph(model_name + ".meta").restore(sess, model_name)

            x = tf.get_default_graph().get_operation_by_name('x').outputs[-1]

            error = tf.get_default_graph().get_operation_by_name('adv_error').outputs[-1]
        else:
            x = tf.placeholder("float64", [1, input_len, xdims], name="x")
            x_exp = tf.expand_dims(x, 0)
            x_patches = tf.extract_image_patches(images=x_exp, ksizes=[1, 1, c.seq_len, 1],
                                                 strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
            x_patches = tf.squeeze(x_patches)
            x_patches = tf.reshape(x_patches, (-1, c.seq_len, xdims))

            # restore and bind
            tf.train.import_meta_graph(model_name + ".meta", input_map={"x:0": x_patches}).restore(sess, model_name)
            out_patches = tf.get_default_graph().get_operation_by_name("out").outputs[-1]
            out_reordered = out_patches[:, -1, :]

            out_restored = tf.reshape(out_reordered, (1, -1, xdims))

            diff = tf.subtract(
                x[:, -out_restored.shape[-2].value:, :],
                out_restored[:, :, :])

            # Using reduce_sum instead of reduce_mean provides larger gradients
            error = tf.reduce_mean(tf.square(diff)) + tf.reduce_max(tf.abs(diff))  # TRY adding reduce_max

        # add gradients to the graph
        t_grad_x = tf.gradients(error, x)

        fetch = [t_grad_x, error]

        loop = tqdm(range(c.max_adv_iter))
        for i in loop:
            # Get the gradient, the full error on the current poison
            res = sess.run(fetch, feed_dict={"x:0": x_poison})

            grad_x = res[0][0]
            full_error = res[1]

            # Check if the error with the current poison is decreasing
            eps = -0.000001
            prev_adv_lr = adv_lr
            # just roll back to the previous working poison and recalculate with it using smaller lr.
            if np.sum([full_error - el < eps for el in prev_adv_errors]) == 0 and adv_lr > 0.00001:
                failed_updates += 1
                # speed up the step decreasing if we keep on failing
                adv_lr = prev_adv_lr * (0.90 ** failed_updates)
                x_poison = prev_x_poison.copy()
            elif adv_lr <= 0.00001:
                if not c.silent: print(
                        "The errors keep growing (%f) and the learning rate (%f) is already small, exiting!"
                        % (full_error, adv_lr))
                break
            else:  # we are good
                # push the error to the queue
                prev_adv_errors.appendleft(full_error)
                failed_updates = 0
                prev_x_poison = x_poison.copy()
                # restore adv_lr
                adv_lr = orig_adv_lr

            # I want to minimize the error thus -
            # Calculate the max of the absolute gradient values.
            # This is used to calculate the step-size.
            grad_absmax = np.abs(grad_x).max()

            # If the gradient is very small then use a lower limit,
            # because we will use it as a divisor.
            if grad_absmax < 1e-10:
                grad_absmax = 1e-10

            step_size = adv_lr / grad_absmax

            x_poison -= step_size * grad_x

    # Final result
    train_err, test_err, n_out, alerts, _, max_err, _ = train_and_test_adv(model_name, 0, 0, x_train,
                                                                           [], x_poison,
                                                                           x_poison, c, False, not c.silent,
                                                                           title="Final test")

    if not c.silent:
        print("Final poisoned training error %f, test error %f, %d alerts, max_err %f" % (
            train_err, test_err, alerts, max_err))
        for col in range(n_out.shape[-1]):
            plt.figure(figsize=(20, 10))
            plt.plot(x_att[:, col], label="Original Attack Input", color='b')
            plt.plot(x_poison.reshape((-1, x_poison.shape[-1]))[:, col], label="Closest Starting Poison Found",
                     color='r')
            plt.legend()
            plt.title("Poison for Column %d" % col)

    return x_poison.reshape(x_att.shape), alerts


# Synthetic tests
def run_synthetic_test(conf):
    start_time = time.time()
    model_name = "./tmp/synthetic_signal_seq_%d_batches_%d" % (conf["seq_len"], conf["batches"])
    # remove old intermediate models
    for f in glob.glob("." + os.sep + "tmp" + os.sep + model_name + '_*.*'):
        os.remove(f)

    np.random.seed(1)
    noise = np.random.normal(0, 0.02, (conf["train_points"], conf["total_len"], 1))
    src, _ = conf["generator"](munchify(conf))

    # TODO - apply proportionally?
    noised_src = noise + src

    model_signal(conf, model_name, noised_src)
    if conf.get("sec_seq_len"):
        tmp = conf["seq_len"]
        sec_model_name = "./tmp/synthetic_signal_seq_%d_batches_%d" % (conf["sec_seq_len"], conf["batches"])
        conf["sec_model_name"] = sec_model_name
        conf["seq_len"] = conf["sec_seq_len"]
        model_signal(conf, sec_model_name, noised_src)
        conf["seq_len"] = tmp

    print(conf)
    start_poison, x_att = generate_poison_and_attack(conf)

    if conf["find_poison"]:
        start_poison, alerts = find_start_poison(conf, model_name, x_att, noised_src)
        if alerts:
            print("Failed to find start poison. Test configuration", conf, "\nResults: ", alerts, " alerts")
            return

    poison_func = poison_model_naive if conf["naive"] else poison_model

    alerts, poison_points, iterations, poisoned_model_file_name = poison_func(conf, model_name, start_poison, x_att,
                                                                              noised_src)
    print("Test configuration", conf, "\nResults: ", alerts,
          " alerts", poison_points, "poison_points", iterations, "iterations")

    time_taken = time.time() - start_time
    print("Took ", time_taken)
    return alerts, poison_points, iterations, time_taken


def synthetic_test(conf):
    points_per_period = 500

    cols = list(conf.keys()) + ['alerts', 'poison_points', 'iterations', 'time']
    df_results = pd.DataFrame(columns=cols)
    sns.set(rc={"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "legend.fontsize": 12}, style="darkgrid")

    failed = 0
    for _ in range(5):
        conf["total_len"] = conf["periods"] * points_per_period
        if conf["single_sequence"]:
            conf["seq_len"] = conf["total_len"]
        conf["batches"] = conf["train_points"]
        conf["optimizer"] = tf.train.GradientDescentOptimizer(conf["lr"])
        res = conf.copy()
        res['alerts'], res['poison_points'], res['iterations'], res[
            'time'] = run_synthetic_test(conf)
        df_results = df_results.append(pd.DataFrame([res]), ignore_index=True)
        df_results.to_csv(datetime.datetime.now().strftime(
            data_path + "PoisoningTests_%d_%m_%Y_%H_%M_%S.csv"))

        if res['alerts']:
            failed += 1
            if failed > 2:
                break

    df_results.to_csv(datetime.datetime.now().strftime(data_path + "PoisoningTests_%d_%m_%Y_%H_%M_%S.csv"))


def read_swat_data_file(file_name):
    df_train_orig = pd.read_csv(data_path + file_name, parse_dates=['Timestamp'], index_col=0, dayfirst=True)
    df_train_orig = df_train_orig.ffill()
    df_train_orig.columns = df_train_orig.columns.str.strip()
    df_train_orig.index = pd.to_datetime(df_train_orig.index, format='%d/%m/%Y %I:%M:%S %p')
    return df_train_orig


def read_swat():
    df_train_orig = read_swat_data_file("SWaT_Dataset_Normal_sub5.csv")
    df_test_orig = read_swat_data_file("SWaT_Dataset_Attack_sub5.csv")

    # removing bad fields dramatically increases the recall rate
    sensor_cols = [col for col in df_train_orig.columns if
                   col not in ['Timestamp', 'Normal/Attack', "AIT201", 'AIT202', 'AIT203', 'P201', 'AIT501', 'AIT503',
                               'AIT504', 'AIT402']]

    # scale sensor data
    scaler = MinMaxScaler((-0.5, 0.5))
    # cut the first unstable part
    df_train_orig = df_train_orig.iloc[85000 // 5:, :]

    X = pd.DataFrame(index=df_train_orig.index, columns=sensor_cols,
                     data=scaler.fit_transform(df_train_orig[sensor_cols]))
    Y = pd.DataFrame(index=df_test_orig.index, columns=sensor_cols,
                     data=scaler.transform(df_test_orig[sensor_cols]))

    attacks = dict()

    # 3
    attacks[3] = dict(
        attack_start='12/28/2015  11:22:00 AM',
        attack_end='12/28/2015  11:28:22 AM',
        field_names=['LIT101', "FIT101", "MV101", "P101"],
        #     delta_hours = 0.6
    )
    # 7
    attacks[7] = dict(
        attack_start='12/28/2015  12:08:25 PM',
        attack_end='12/28/2015  12:15:33 PM',
        field_names=['LIT301', "FIT201", 'P302'],
        #     delta_hours = 1.15
    )
    # Attack 8: 28/12/2015 13:10:10	13:26:13
    attacks[8] = dict(
        attack_start='12/28/2015  01:10:10 PM',
        attack_end='12/28/2015  01:26:13 PM',
        field_names=['DPIT301', "P602"])

    # Attack 10 28/12/2015 14:16:20	14:19:00
    attacks[10] = dict(
        attack_start='12/28/2015  02:16:20 PM',
        attack_end='12/28/2015  02:19:00 PM',
        field_names=['FIT401', "P402"])

    # Attack 16 29/12/2015 11:57:25	12:02:00
    attacks[16] = dict(
        attack_start='12/29/2015  11:57:21 AM',
        attack_end='12/29/2015  12:02:00 PM',
        field_names=['LIT301', "FIT201", 'P302'])  # adding more related fields allows better model

    # 31 31/12/2015 22:05:34	22:11:40
    attacks[31] = dict(
        attack_start='12/31/2015  10:05:34 PM',
        attack_end='12/31/2015  10:11:40 PM',
        field_names=["LIT401", 'P302', 'P402', 'LIT301'])  # adding more related fields allows better model

    # 32 1/01/2016 10:36:00	10:46:00 LIT-301
    attacks[32] = dict(
        attack_start='01/01/2016  10:36:00 AM',
        attack_end='01/01/2016  10:46:00 AM',
        field_names=['LIT301', "FIT201", 'P302'])

    # #33 1/01/2016 14:21:12	14:28:35	LIT-101
    attacks[33] = dict(
        attack_start='01/01/2016  02:21:12 PM',
        attack_end='01/01/2016  02:28:35 PM',
        field_names=['LIT101', "FIT101", "MV101", "P101"])
    # field_names = sensor_cols[:5]

    # 36 1/01/2016 22:16:01	22:25:00	LIT-101
    attacks[36] = dict(
        attack_start='01/01/2016  10:16:01 PM',
        attack_end='01/01/2016  10:25:00 PM',
        field_names=['LIT101', "FIT101", "MV101", "P101"])

    return X, Y, attacks


# # Synthetic SWaT attacks
# I want to test synthetic attacks on the SWaT data, namely generate the attack on the top of a real SWaT signal
# I arrived to the conclusion that it is not feasible to find the "correct" good starting poison from the given data.
# Therefore I should synthesize attacks similar to the SWaT ones and test them.
#
# | Attack | Imitation | Notes |
# | ------ | --------- | ------|
# | 3 | requires linear increase of LIT101 to 0.824721 | |
# | 7 | LIT301 fixed at 1.309346 | |
# | 16 | LIT301 linearly decreases to -1.393334 | |
# | 31 | LIT401 fixed at -1.269128 | |
# | 32 | LIT301 fixed at 1.313708 | |
# | 33 | LIT101 fixed at -1.081831 | |
# | 36 | LIT101 fixed at -1.232907| |
#

def run_synthetic_swat_test(conf, x_train_full, extended_att_len):
    start_time = time.time()
    model_name = "./tmp/synthetic_swat_seq_%d" % (conf["seq_len"])
    # remove old intermediate models
    for f in glob.glob("." + os.sep + "tmp" + os.sep + model_name + '_*.*'):
        os.remove(f)

    np.random.seed(1)
    conf["total_len"] = conf["train_points"] * extended_att_len
    x_train = x_train_full[-conf["total_len"]:]
    period_start = conf["period_start"] if "period_start" in conf else 0
    # it will be used for generating a clean validation input
    conf["generator"] = EchoGenerator(x_train_full[period_start:extended_att_len])

    model_signal(conf, model_name, x_train_full)

    start_poison, x_att = generate_poison_and_attack(conf)
    if conf["find_poison"]:
        start_poison, alerts = find_start_poison(conf, model_name, x_att, x_train)
        if alerts:
            print("Failed to find start poison. Test configuration", conf, "\nResults: ", alerts, " alerts")
            return alerts, 0, 0, 0

    poison_func = poison_model_naive if conf["naive"] else poison_model
    alerts, poison_points, iterations, poisoned_model_file_name = poison_func(conf, model_name, start_poison, x_att,
                                                                              x_train)

    print("Test configuration", conf, "\nResults: ", alerts,
          " alerts", poison_points, "poison_points", iterations, "iterations")

    time_taken = time.time() - start_time
    print("Took ", time_taken)
    return alerts, poison_points, iterations, time_taken


def syn_swat_test(conf, att_num=None):
    X, Y, attacks = read_swat()
    syn_attacks = dict()
    syn_attacks[3] = dict(
        field_names=['LIT101', "FIT101", "MV101", "P101"],  # sensor_cols#
        extended_att_len=1100,  # total length of the attack input, taken from the length of attack 33
        att_len=77,
        att_start=700,
        att_magn=0.33,
        att_point="CUSTOM_LINEAR",
        period_start=200
    )

    syn_attacks[33] = dict(  # in reality, "FIT101" and "MV101" are stopped as the tank seems full
        field_names=['LIT101', "FIT101", "MV101", "P101"],  # sensor_cols#
        extended_att_len=900,  # total length of the attack input, taken from the length of attack 33
        att_len=88,
        att_start=450,
        att_magn=1,
        att_point="CUSTOM_FIXED",
        period_start=0
    )

    syn_attacks[36] = dict(
        field_names=['LIT101', "FIT101", "MV101", "P101"],  # sensor_cols#
        extended_att_len=1400,  # total length of the attack input, taken from the length of attack 33
        att_len=106,
        att_start=260,
        att_magn=-1,
        att_point="CUSTOM_FIXED",
        period_start=0
    )

    syn_attacks[31] = dict(
        field_names=["LIT401", 'P302', 'LIT301', "P402"],
        extended_att_len=1100,  # total length of the attack input, taken from the length of attack 33
        att_len=76,
        att_start=800,
        period_start=100,
        att_magn=-1,
        att_point="CUSTOM_FIXED"
    )

    syn_attacks[7] = dict(  # in reality FIT201 is stopped as the tank seems full
        field_names=['LIT301', "FIT201", 'P302'],  # sensor_cols#
        extended_att_len=1400,  # total length of the attack input, taken from the length of attack 33
        att_len=86,
        att_start=475,
        att_magn=1.0,
        att_point="CUSTOM_FIXED",
        period_start=500
    )

    # att 16
    syn_attacks[16] = dict(  # in reality, FIT201 is stopped as th tank seems full
        field_names=['LIT301', "FIT201", 'P302'],  # sensor_cols#
        extended_att_len=1400,  # total length of the attack input, taken from the length of attack 33
        att_len=56,
        att_start=536,
        att_magn=-0.5,
        att_point="CUSTOM_LINEAR",
        period_start=500
    )

    # att 32
    syn_attacks[32] = dict(  # in reality, FIT201 is started as the tank seems empty
        field_names=['LIT301', "FIT201", 'P302'],  # sensor_cols#
        extended_att_len=1200,  # total length of the attack input, taken from the length of attack 33
        att_len=120,
        att_start=570,
        att_magn=1,
        att_point="CUSTOM_FIXED",
        period_start=300
    )

    cols = list(conf.keys()) + ['alerts', 'poison_points', 'iterations', 'time']
    df_results = pd.DataFrame(columns=cols)
    if att_num is None:
        attacks_list = list(syn_attacks.keys())
    else:
        attacks_list = [att_num]
    for at in attacks_list:
        print("Attack", at)
        field_names = syn_attacks[at]["field_names"]
        extended_att_len = syn_attacks[at]["extended_att_len"]
        conf["att_len"] = syn_attacks[at]["att_len"]
        conf["att_start"] = syn_attacks[at]["att_start"]
        conf["att_magn"] = syn_attacks[at]["att_magn"]
        conf["att_point"] = syn_attacks[at]["att_point"]
        conf["period_start"] = syn_attacks[at]["period_start"] if "period_start" in syn_attacks[at] else 0

        if at in [3, 16, 31]:
            conf["threshold"] = 0.1
        else:
            conf["threshold"] = 0.2

        x_train_full = X[field_names].values

        failed = 0
        for _ in range(5):
            conf["total_len"] = conf["train_points"] * extended_att_len
            res = conf.copy()
            res['alerts'], res['poison_points'], res['iterations'], res['time'] = \
                run_synthetic_swat_test(conf,
                                        x_train_full,
                                        extended_att_len)
            df_results = df_results.append(pd.DataFrame([res]), ignore_index=True)
            # add to the test results!
            timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            df_results.to_csv(data_path + "PoisoningTests_swat_syn_%d_%s.csv" % (at, timestamp))
            if res['alerts']:
                failed += 1
                if failed > 2:
                    break

    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    df_results.to_csv(data_path + "PoisoningTests_swat_syn_%s.csv" % timestamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test', choices=['syn', 'swat'],
                        help="Run training and test for selected data. "
                             "Specify either syn or swat. "
                             "See the synthetic_test and syn_swat_test for multiple test parameters")
    parser.add_argument("-c", "--configuration", help="A dictionary with test parameters (without .py). "
                                                      "Defaults to conf_syn or conf_swat correspondingly.")

    parser.add_argument("-a", "--attack", type=int, choices=[3, 7, 16, 31, 32, 33, 36],
                        help="SWaT attack to poison")

    args = parser.parse_args()
    conf = None  # will be overwritten in the import
    if args.configuration is None:
        args.configuration = "conf_syn" if args.test == 'syn' else "conf_swat"

    conf_mod = importlib.import_module(args.configuration)

    if args.test == 'syn':
        synthetic_test(conf_mod.conf)
    if args.test == 'swat':
        if args.attack is not None:
            syn_swat_test(conf_mod.conf, args.attack)
        else:
            syn_swat_test(conf_mod.conf)
