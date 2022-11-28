import tensorflow as tf

def dice_coefficient( y_true_cls, y_pred_cls, training_mask):
    """
    Compute Dice loss from Sorensen-Dice coefficient. See Eq. (1) Weinman et al.
    (ICDAR 2019) and Milletari et al. (3DV 2016).

    Parameters
      y_true_cls   : binary ground truth score map (1==text, 0==non-text), 
                       [batch_size, tile_size/4, tile_size/4, 1]
      y_pred_cls   : predicted score map in range (0,1), same size as y_true_cls
      training_mask: binary tensor to indicate which locations should be included 
                       in the calculation, same size as y_true_cls
    Returns
      loss: scalar tensor between 0 and 1
    """
    eps = 1e-5 # added for numerical stability
    intersection = tf.math.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.math.reduce_sum( tf.square(y_true_cls) * training_mask) + \
            tf.math.reduce_sum( tf.square(y_pred_cls) * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    #tf.summary.scalar('classification_dice_loss', loss, family='train/losses')
    return loss


def total_loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    # NOTE: Gerasimos changed the name of the function from 'loss' to 'total_loss
    '''
    Compute total loss as the weighted sum of score loss (given by a
    Dice loss), rbox loss (defined as an IoU loss), and angle loss
    (i.e., cosine loss).  See Eq. (6) in Weinman et al. (ICDAR 2019).

    Parameters
     y_true_cls   : binary ground truth score map (1==text, 0==non-text), 
                    [batch_size,tile_size/4,tile_size/4, 1]
     y_pred_cls   : predicted score map in range (0,1), same size as y_true_cls
     y_true_geo   : ground truth box geometry map with shape 
                      [batch_size,tile_size/4,tile_size/4, 5]
     y_pred_geo   : predicted box geometry map, same size as y_true_geo
     training_mask: binary tensor to indicate which locations should be included 
                      in loss the calculations, same size as y_true_cls

    Returns
     total_loss: a scalar

    '''
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(
        value=y_true_geo,
        num_or_size_splits=5,
        axis=3)
    
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(
        value=y_pred_geo,
        num_or_size_splits=5,
        axis=3)
    
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    
    w_union = tf.math.minimum(d2_gt, d2_pred) + tf.math.minimum(d4_gt, d4_pred)
    h_union = tf.math.minimum(d1_gt, d1_pred) + tf.math.minimum(d3_gt, d3_pred)
    
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    
    L_AABB = -tf.math.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.math.cos(theta_pred - theta_gt)
    
    #tf.summary.scalar('geometry_AABB',
                      #tf.reduce_mean(L_AABB * y_true_cls * training_mask),
                      #family='train/losses')
    #tf.summary.scalar('geometry_theta',
                      #tf.reduce_mean(L_theta * y_true_cls * training_mask),
                      #family='train/losses')
    
    L_g = L_AABB + 20 * L_theta

    total_loss = tf.math.reduce_mean(L_g * y_true_cls * training_mask) \
                 + classification_loss

    return total_loss