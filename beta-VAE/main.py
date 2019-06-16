# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os

import matplotlib.pyplot as plt

from vae import Vae
from data_load import DataLoad

tf.app.flags.DEFINE_integer("epoch_size", 2000, "epoch size")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.app.flags.DEFINE_float("gamma", 100.0, "gamma param for latent loss")
tf.app.flags.DEFINE_float("capacity_limit", 20.0,
                          "encoding capacity limit param for latent loss")
tf.app.flags.DEFINE_integer("capacity_change_duration", 100000,
                            "encoding capacity change duration")
tf.app.flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory")
tf.app.flags.DEFINE_string("log_file", "./log", "log file directory")
tf.app.flags.DEFINE_boolean("training", True, "training or not")

flags = tf.app.flags.FLAGS

def disres(sess,model,dset,saver):
    summary_writer = tf.summary.FileWriter(flags.log_file, sess.graph)
    n_samples = dset.sample_size
    recs_check_images = dset.get_random_images(10)
    indices = list(range(n_samples))

    step = 0
    # Training cycle
    for epoch in range(flags.epoch_size):
        # Shuffle image indices
        random.shuffle(indices)

        avg_cost = 0.0
        total_batch = n_samples // flags.batch_size

        # Loop over all batches
        for i in range(total_batch):
            # Generate image batch
            batch_indices = indices[flags.batch_size*i : flags.batch_size*(i+1)]
            batch_xs = dset.get_images(batch_indices)

            # Fit training using batch data
            reconstr_loss, latent_loss, summary_str = model.partial_fit(sess, batch_xs, step)
            summary_writer.add_summary(summary_str, step)
            step += 1

        # Image reconstruction check
        recs_check(sess, model, recs_check_images)

        # Disentangle check
        dise_check(sess, model, dset)

        # Save checkpoint
        saver.save(sess, flags.checkpoint_dir + '/' + 'checkpoint', global_step = step)


def recs_check(sess, model, images):
    # Check image reconstruction
    x_reconstruct = model.reconstruct(sess, images)
    if not os.path.exists("recs_img"):
        os.mkdir("recs_img")

    for i in range(len(images)):
        org_img = images[i].reshape(64, 64)
        org_img = org_img.astype(np.float32)
        recs_img = x_reconstruct[i].reshape(64, 64)
        plt.imsave("recs_img/{0}_orig.png".format(i),org_img)
        plt.imsave("recs_img/{0}_recs.png".format(i), recs_img)


def dise_check(sess, model, dset, save_original=False):
    img = dset.get_image(shape=1, scale=2, orientation=5)
    if save_original:
        plt.imsave("original.png", img.reshape(64, 64).astype(np.float32))

    batch_xs = [img]
    z_mean, z_log_sigma_sq = model.transform(sess, batch_xs)
    z_sigma_sq = np.exp(z_log_sigma_sq)[0]

    # Print variance
    zss_str = ""
    for i,zss in enumerate(z_sigma_sq):
        str = "z{0}={1:.4f}".format(i,zss)
        zss_str += str + ", "
    print(zss_str)

    # Save disentangled images
    z_m = z_mean[0]
    n_z = 10

    if not os.path.exists("dise_img"):
        os.mkdir("dise_img")

    for target_z_index in range(n_z):
        for ri in range(n_z):
            value = -3.0 + (6.0 / 9.0) * ri
            z_mean2 = np.zeros((1, n_z))
            for i in range(n_z):
                if( i == target_z_index ):
                    z_mean2[0][i] = value
                else:
                    z_mean2[0][i] = z_m[i]
            recs_img = model.generate(sess, z_mean2)
            rimg = recs_img[0].reshape(64, 64)
            plt.imsave("dise_img/check_z{0}_{1}.png".format(target_z_index,ri), rimg)


def load_checkpoints(sess):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
    else:
        print("Could not find old checkpoint")
        if not os.path.exists(flags.checkpoint_dir):
            os.mkdir(flags.checkpoint_dir)
    return saver


def main(argv):
    file_name = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    dset=DataLoad(file_name)

    sess = tf.Session()

    model = Vae(gamma=flags.gamma,
              capacity_limit=flags.capacity_limit,
              capacity_change_duration=flags.capacity_change_duration,
              learning_rate=flags.learning_rate)

    sess.run(tf.global_variables_initializer())

    saver = load_checkpoints(sess)

    if flags.training:
        disres(sess, model, dset, saver)
    else:
        recs_check_images = dset.get_random_images(10)
        # Image reconstruction check
        recs_check(sess, model, recs_check_images)
        # Disentangle check
        dise_check(sess, model, dset)


if __name__ == '__main__':
    tf.app.run()
