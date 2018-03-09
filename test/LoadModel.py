import tensorflow as tf
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp

g_model_path = './model'
g_meta_path = g_model_path+'/dnn.ckpt-200.meta'
g_check_path= g_model_path+'/dnn.ckpt-200'

sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(g_meta_path)
saver.restore(sess,tf.train.latest_checkpoint(g_model_path))
# chkp.print_tensors_in_checkpoint_file(g_check_path, tensor_name='', all_tensors=True,all_tensor_names=True)


# ** Get Variables
print(sess.graph.collections)
print(' - - - -trainable_variables')
all_vars = sess.graph.get_collection('trainable_variables')
for v in all_vars:
    print(v.name)
    print(sess.run(v))
print(' - - - - - - - - - - - - - - - ')
print(' - - - -train_op')
all_vars = sess.graph.get_collection('train_op')
for v in all_vars:
    print(v.name)
    # print(sess.run(v))
print(' - - - - - - - - - - - - - - - ')


# ** Tmp input data
data = np.loadtxt('./data.csv', delimiter=',',
                  unpack=True, dtype='float32')

# x_data = np.transpose(data[0:2])
# y_data = np.transpose(data[2:])
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# ** Get Models and Insert prediction images
operations = sess.graph.get_operations()
for op in operations:
    if op.name.endswith('Relu'):
        print('Op: ' + op.name)
        print(op.outputs[0].inputs)
        # print(sess.run(op.outputs))
        print(sess.run(op.outputs[0], feed_dict={X:x_data}))

# ** Tmp node
# for node in sess.graph_def.node:
#     if node.name.endswith('Relu'):
#         print('Node : ' + node.name)
#     # print(node.attr['value'].tensor)

# operations = sess.graph.get_operation_by_name('relu')
print('tmp')



