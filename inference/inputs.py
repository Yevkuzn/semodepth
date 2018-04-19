# Copyright 2017. All rights reserved.
# Computer Vision Group, Visual Computing Institute
# RWTH Aachen University, Germany
# This file is part of the semodepth project.
# Authors: Yevhen Kuznietsov (yevkuzn@gmail.com OR Yevhen.Kuznietsov@esat.kuleuven.be)
# semodepth project is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or any later version.
# semodepth project is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.


import tensorflow as tf
import config as cfg

def process_record(in_path, out_path):
    """Reads input image and maps input file name to it
    Args:
    in_path: path to the file containing where the input image is stored.
    out_path: output path.

    Returns:
    img: 3-D Tensor of [cfg.FLAGS.inference_image_height, cfg.FLAGS.inference_image_width,
    cfg.FLAGS.inference_image_channels] of type float32, input image.
    out_path: output path.
    """

    # Read input image
    img_contents = tf.read_file(in_path)
    # Decode raw file to png
    img = tf.cast(tf.image.decode_png(img_contents, channels=cfg.FLAGS.input_image_channels), dtype=tf.float32)
    # Resize image to match single pipeline input size
    img = tf.image.resize_images(img, [cfg.FLAGS.inference_image_height, cfg.FLAGS.inference_image_width],
                                 method=tf.image.ResizeMethod.BILINEAR)

    return img, out_path

def read_filenames(filenames_path):
    """Read lists of input and output file names
    Args:
    filenames_path: path to the file containing the pairs of input image names and names of files to save predictions to.

    Returns:
    l_in: list of input names.
    l_out: list of output names.
    """

    f = open(filenames_path, 'r')
    l_in = []
    l_out = []
    for line in f:
        try:
            i_name, o_name = line[:-1].split(',')
        except ValueError:
            i_name = o_name = line.strip("\n")
            print("Something is wrong with the filelist")

        if not tf.gfile.Exists(i_name):
            raise ValueError('Failed to find file: ' + i_name)

        l_in.append(i_name)
        l_out.append(o_name)

    return l_in, l_out

def generate_data_iterator(filenames_path):
    """Generates iterator to data
    Args:
    filenames_path: path to the file containing the pairs of input image names and names of files to save predictions to.

    Returns:
    data_iterator: iterator to image (4D tensor) - output name pairs.
    iterator_init_op: operation, initializing iterator.
    """
    #Read file name lists
    img, savepath = read_filenames(filenames_path)
    #Convert to tensors
    img = tf.constant(img)
    savepath = tf.constant(savepath)
    #Create a dataset object
    data = tf.data.Dataset.from_tensor_slices((img, savepath))
    #Map image names to image contents
    data = data.map(process_record)
    #Transform input into batched form
    data = data.batch(1)
    #Create the iterator to mapped pairs
    data_iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
    iterator_init_op = data_iterator.make_initializer(data)

    return data_iterator, iterator_init_op
