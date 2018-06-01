from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf



exported_path = "/home/xavier/Desktop/developpement/Network/gesture_detection/model/1516908729/"

with tf.Session() as sess:
   #load the saved model
   tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], exported_path)
   #Prepare model input, the model expects a float array to be passed to x
   # check line 28 serving_input_receiver_fn
   model_input= tf.train.Example(features=tf.train.Features(feature={
              'x': tf.train.Feature(float_list=tf.train.FloatList(value=[-3,-10,-13,-23,-32,-40,-34,-9,11,36.5,29.5,30,45,44,40.25,8.75,-32,-27.5,-25,-24,-19,-8,12,24,22,5,-9,-15,-16,-13.5,-17.5,-4]))        
              })) 

   #get the predictor , refer tf.contrib.predicdtor
   predictor= tf.contrib.predictor.from_saved_model(exported_path)

   #get the input_tensor tensor from the model graph
   # name is input_tensor defined in input_receiver function refer to tf.dnn.classifier
   input_tensor=tf.get_default_graph().get_tensor_by_name("input_tensors:0")
   #get the output dict
   # do not forget [] around model_input or else it will complain shape() for Tensor shape(?,)
   # since its of shape(?,) when we trained it
   model_input=model_input.SerializeToString()
   output_dict= predictor({"inputs":[model_input]})
   print(" prediction is " , output_dict['scores'])
