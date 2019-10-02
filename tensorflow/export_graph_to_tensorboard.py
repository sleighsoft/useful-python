# Requires
# Python, Tensorflow (2.0+)
#
# Can be installed with:
# pip install tensorflow
#
# Description:
# Reads a protobuf file and exports the graph to `output_directory` so it can
# be visualized in tensorboard.

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.summary import summary

output_directory = ''
graph_def_pb_file = "<file>.pb"

with tf.io.gfile.GFile(graph_def_pb_file, "rb") as f:
    graph_def = graph_pb2.GraphDef()
    graph_def.ParseFromString(f.read())


with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")

    pb_visual_writer = summary.FileWriter(output_directory)
    pb_visual_writer.add_graph(graph)
