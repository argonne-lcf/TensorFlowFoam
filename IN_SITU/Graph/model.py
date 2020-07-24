import tensorflow as tf
from tensorflow import keras

# Batch of input (1x9 matrices)
# Batch of output (1x1) matrices
x = tf.placeholder(tf.float32, shape=[None, 1, 9], name='input')
y = tf.placeholder(tf.float32, shape=[None, 1, 1], name='target')

# Trivial model
h = tf.layers.dense(x, 30, activation='tanh',kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=1.0))
y_ = tf.identity(tf.layers.dense(h, 1, kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=1.0)), name='output')

# # Using Keras API
# reshaped_input = keras.layers.Reshape((2,))(x)
# hidden_layer_1 = keras.layers.Dense(30,activation='tanh')(reshaped_input)
# output = keras.layers.Dense(3,activation='linear')(hidden_layer_1)
# reshaped_output = keras.layers.Reshape((1,3))(output)

# # Optimize loss
loss = tf.reduce_mean(tf.square(y_ - y), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss, name='train')

init = tf.initialize_all_variables()

# for op in keras.backend.get_session().graph.get_operations():
#     print(op.name)
# exit()

# tf.train.Saver.__init__ adds operations to the graph to save
# and restore variables.
saver_def = tf.train.Saver().as_saver_def()

print('Run this operation to initialize variables     : ', init.name)
print('Run this operation for a train step            : ', train_op.name)
print('Feed this tensor to set the checkpoint filename: ', saver_def.filename_tensor_name)
print('Run this operation to save a checkpoint        : ', saver_def.save_tensor_name)
print('Run this operation to restore a checkpoint     : ', saver_def.restore_op_name)

# Write the graph out to a file.
# with open('graph.pb', 'w') as f:
  # f.write(tf.get_default_graph().as_graph_def().SerializeToString())
session = keras.backend.get_session()
tf.train.write_graph(session.graph, './', 'graph.pb', as_text=False)