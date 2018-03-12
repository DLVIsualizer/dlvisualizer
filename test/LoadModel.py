import tensorflow as tf
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp

g_model_path = './model'
g_meta_path = g_model_path+'/dnn.ckpt-100.meta'
g_check_path= g_model_path+'/dnn.ckpt-100'

# ** First let's load meta graph and restore weights
sess=tf.Session()
saver = tf.train.import_meta_graph(g_meta_path)
saver.restore(sess,tf.train.latest_checkpoint(g_model_path))

# ** Get Variables
print(sess.graph.collections)
print(' - - - -trainable_variables- - - - - - - - - - - - -')
all_vars = sess.graph.get_collection('trainable_variables')
for v in all_vars:
    print(v.name)
    print(sess.run(v))
print(' - - - - - - - - - - - - - - -  - - - --  - -')
print(' - - - -train_op- - - - - - - - - - - - - - - - - -')
all_vars = sess.graph.get_collection('train_op')
for v in all_vars:
    print(v.name)
    # print(sess.run(v))
print(' - - - - - - - - - - - - - - -  - - - - -- - - -  --')


# ** Tmp input data

PlaceX = sess.graph.get_tensor_by_name('PlaceX:0')
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])
x_data0 = x_data[1,:]
x_data0 = np.array([x_data0])

# ** Get Models and Insert prediction images
print(' - - - -Activation Operations - - - - - - - - - - - - - - - - - -')
operations = sess.graph.get_operations()
for op in operations:
    if op.name.endswith('Relu'):
        print('Op: ' + op.name)
        print(op.outputs)
        print(sess.run(op.outputs, feed_dict={PlaceX:x_data0}))

print(' - - - - - - - - - - - - - - -  - - - - -- - - -  --')



