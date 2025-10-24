import tensorflow as tf
import keras.backend as K
from keras import losses


# def sparse_categorical_cross_entropy_pos_contrib(target_rank, pred_rank):
#     target_class_ids = tf.reshape(target_rank, (-1,))

#     # Only positive ROIs contribute to the loss. And only
#     # the right class_id of each ROI. Get their indices.
#     positive_ix = tf.where(target_class_ids > 0)[:, 0]

#     # Gather the ranks (predicted and true) that contribute to loss
#     y_true = tf.gather(target_rank, positive_ix, axis=1)
#     y_pred = tf.gather(pred_rank, positive_ix, axis=1)

#     loss = K.switch(tf.size(y_true) > 0,
#                     losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred),
#                     tf.constant(0.0))

#     loss = K.mean(loss)

#     return loss

def sparse_categorical_cross_entropy_pos_contrib(target_rank, pred_rank, from_logits=True, eps=1e-7):
    # target_rank: [B, N] with 0 = non-positive, >0 = class id
    # pred_rank:   [B, N, C] class scores (logits if from_logits=True)
    y_true = tf.cast(target_rank, tf.int32)                  # [B, N]
    mask  = tf.cast(y_true > 0, pred_rank.dtype)            # [B, N]

    # CE per (B,N); Keras will broadcast y_true over last dim of pred_rank
    ce = tf.keras.losses.sparse_categorical_crossentropy(
        y_true=y_true, y_pred=pred_rank, from_logits=from_logits
    )                                                       # [B, N]

    # keep only positives
    ce = ce * mask                                          # [B, N]

    denom = tf.reduce_sum(mask)                             # scalar
    loss  = tf.where(denom > 0,
                     tf.reduce_sum(ce) / (denom + eps),
                     tf.zeros([], dtype=pred_rank.dtype))
    return loss





