import tensorflow as tf


def minus_IoU(mask, y_pred):
    #print("des:", desired_out)
    #print("pred:", pred_out)
    #assert desired_out.shape == pred_out.shape
    	
    intersection = tf.reduce_sum(tf.boolean_mask(y_pred, mask))
    union = tf.math.count_nonzero(mask, dtype=tf.dtypes.float32) + tf.reduce_sum(y_pred)

    return - 2 * intersection / union
