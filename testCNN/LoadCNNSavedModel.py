import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# Load Model & Tensors(PlaceHolder)
sess=tf.Session()
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
input_key = 'x_input'
keep_prob_key = 'keep_prob'
output_key = 'y_output'

export_path =  './savedmodel'
meta_graph_def = tf.saved_model.loader.load(
           sess,
          [tf.saved_model.tag_constants.SERVING],
          export_path)
signature = meta_graph_def.signature_def

x_tensor_name = signature[signature_key].inputs[input_key].name
keep_prob_tensor_name = signature[signature_key].inputs[keep_prob_key].name
y_tensor_name = signature[signature_key].outputs[output_key].name

X = sess.graph.get_tensor_by_name(x_tensor_name)
KeepProb = sess.graph.get_tensor_by_name(keep_prob_tensor_name)
Y = sess.graph.get_tensor_by_name(y_tensor_name)

# ** Get Variables
print(' - - - -trainable_variables- - - - - - - - - - - - -')
all_vars = sess.graph.get_collection('trainable_variables')
# for v in all_vars:
#     print(v.name)
#     print(sess.run(v))
print(' - - - - - - - - - - - - - - -  - - - --  - -')
print(' - - - -train_op- - - - - - - - - - - - - - - - - -')
all_vars = sess.graph.get_collection('train_op')
# for v in all_vars:
#     print(v.name)
    # print(sess.run(v))
print(' - - - - - - - - - - - - - - -  - - - - -- - - -  --')

# Prepare pyplot
fig = plt.figure(figsize=(64,3))

# ** Get Models and Insert prediction images
print(' - - - -Activation Operations - - - - - - - - - - - - - - - - - -')
operations = sess.graph.get_operations()

# Dictionary of network operations (activations) to fetch
fetches = {}

# Manual inputs using placeholder 'images' of shape [N,H,W,C]
feed_dict ={}
feed_dict[X] = mnist.test.images[0].reshape(-1,28,28,1)
feed_dict[KeepProb] = 1.0

# Layer Names
layerNames = []

for op in operations:
    if op.name.endswith('Relu'):
        layerNames += [op.name]
        fetches[op.name] = op.outputs

image = sess.run(fetches, feed_dict=feed_dict)
# print(image)

i = 0
for layerName in layerNames:
    if(i == 2):
        break

    imageI = image[layerName]
    num_of_filters = imageI[0].shape[3]

    for k in range(num_of_filters):
        subplot = fig.add_subplot(image.__len__(), 64, 64*i+k+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        imageK = imageI[0][:,:,:,k]
        subplot.imshow(imageK[0])
        print(">>>",i,"layer, ", k,"filter")
        print(imageK)

    i+= 1

plt.show()

print(' - - - - - - - - - - - - - - -  - - - - -- - - -  --')



