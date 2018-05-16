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
import inputs as inputs
import inference_ResNet_ls_eval as inference
import scipy.io as sio
import config as cfg


def predict():
    with tf.Graph().as_default():

        #Graph part for reading images and producing predictions
        data_iterator, iterator_init_op = inputs.generate_data_iterator(cfg.FLAGS.filenames_path)
        next_img, next_savepath = data_iterator.get_next()
        shape = tf.shape(next_img)

        prediction = inference.inference(next_img)
        prediction = cfg.FLAGS.bf / prediction

        #Prepare to restore variables
        saver = tf.train.Saver(tf.global_variables())
        
        #Configure GPU options
        config = tf.ConfigProto(
            device_count={'GPU': 1}
        )
        config.gpu_options.allow_growth = False
        config.gpu_options.allocator_type = 'BFC'

        #Start running operations on the Graph.
        sess = tf.Session(config=config)
        sess.run([tf.global_variables_initializer(), iterator_init_op])

        #Restore trained network
        saver.restore(sess, cfg.FLAGS.chkpt_path)

        while True:
            try:
                #Execute graph
                _prediction, _savename, _sh = sess.run([prediction, next_savepath, shape])
                #Save predicitons as.mat file
                savename = _savename[0]
                sio.savemat(savename, {"mat":_prediction})
            except Exception as e:
                #check if the error is caused by the absense of data to iterate through
                if e.message.find("End of sequence") != -1:
                    print("Finished")
                else:
                    print(str(e))
                break




def main(argv=None):
    predict()

if __name__ == '__main__':
    tf.app.run()
