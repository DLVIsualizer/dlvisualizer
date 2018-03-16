import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

g_model_path = './model'
g_meta_path = g_model_path+'/dnn.ckpt-0.meta'
g_check_path= g_model_path+'/dnn.ckpt-0'

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
keep_prob= sess.graph.get_tensor_by_name('KeepProb:0')

# Prepare pyplot
fig = plt.figure(figsize=(64,3))

# ** Get Models and Insert prediction images
print(' - - - -Activation Operations - - - - - - - - - - - - - - - - - -')
operations = sess.graph.get_operations()
act_ops = []
for op in operations:
    if op.name.endswith('Relu'):
        act_ops += [op]

for i in range(act_ops.__len__()-1):
        op = act_ops[i]
        print('Op: ' + op.name)
        # print(op.outputs)
        image = sess.run(op.outputs, feed_dict={PlaceX:mnist.test.images[0].reshape(-1, 28, 28, 1),keep_prob:1.0})

        num_of_filters = image[0].shape[3]

        for k in range(num_of_filters):
            subplot = fig.add_subplot(act_ops.__len__(), 64, 64*i+k+1)
            subplot.set_xticks([])
            subplot.set_yticks([])
            imageK = image[0][:,:,:,k]
            subplot.imshow(imageK[0])
            # print(imageK)

plt.show()


print(' - - - - - - - - - - - - - - -  - - - - -- - - -  --')



