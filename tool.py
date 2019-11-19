import tensorflow as tf

checkpoint_dir = "./models"
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)