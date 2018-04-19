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

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('input_image_channels', 3, """Number of channels in the input image.""")
tf.app.flags.DEFINE_integer('inference_image_height', 188, """Height of the image for inference.""")
tf.app.flags.DEFINE_integer('inference_image_width', 621, """Width of the image for inference.""")

tf.app.flags.DEFINE_string('filenames_path', "...", """Path to file contatining input-output names.""")
tf.app.flags.DEFINE_string('chkpt_path', "...", """Path to saved model""")

tf.app.flags.DEFINE_float('bf', 359.7176277195809831 * 0.54, """Baseline times focal length""")
