import tensorflow as tf
import os
dir = os.path.dirname(os.path.realpath(__file__))
'''
v1 = tf.Variable(1. , name="v1")
v2 = tf.Variable(2. , name="v2")
# Let's design an operation
a = tf.add(v1, v2)

# We can check easily that we are indeed in the default graph
#print(a.graph == tf.get_default_graph())
# By default, the Saver handles every Variables related to the default graph
#all_saver = tf.train.Saver()
# But you can precise which vars you want to save (as a list) and under which name (with a dict)
#v2_saver = tf.train.Saver({"v2": v2})

# By default the Session handles the default graph and all its included variables
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  all_saver.save(sess, dir + '/data-all')
sess.close()
'''

saver = tf.train.import_meta_graph('data-all.meta')
graph = tf.get_default_graph()
v1 = graph.get_tensor_by_name('v1:0')
v2 = graph.get_tensor_by_name('v2:0')
with tf.Session() as sess:
  graph = tf.get_default_graph()
  saver.restore(sess,'data-all.data-00000-of-00001')
  sess.sun()







