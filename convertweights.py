import tensorflow as tf

meta_path = 'model/LPRtf3.ckpt-10000.meta' # Your .meta file
output_node_names = [n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node]   # Output nodes

with tf.compat.v1.Session() as sess:
    # Restore the graph
    saver = tf.compat.v1.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('model/'))

    # Freeze the graph
    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())