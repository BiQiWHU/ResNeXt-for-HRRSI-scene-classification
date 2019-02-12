# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 20:16:24 2019

@author: Qi_Bi
"""


# In[1]:


# -*- coding: utf-8 -*-
from tfdata import *
import numpy as np
import tensorflow as tf


# In[2]:


def weight_variable(shape, name):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name='weights',
                                  shape=shape,
                                  trainable=True,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        #REGULARIZATION_RATE=0.000001
        #regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        #tf.add_to_collection('losses', regularizer(weights))
        return weights


# In[3]:


def bias_variable(shape, name):
    with tf.variable_scope(name) as scope:
        biases = tf.get_variable(name='biases',
                                 shape=shape,
                                 trainable=True,
                                 initializer=tf.constant_initializer(0.01))

        return biases


# In[4]:


def conv2d(input, in_feature_dim, out_feature_dim, kernel_size, stride, with_bias=True, name=None):
    W = weight_variable([kernel_size, kernel_size, in_feature_dim, out_feature_dim], name=name)
    conv = tf.nn.conv2d(input, W, [1, stride, stride, 1], padding='SAME')
    if with_bias:
        return conv + bias_variable([out_feature_dim], name=name)
    return conv


# In[7]:


def avg_pool(input, s, stride):
    return tf.nn.avg_pool(input, [1, s, s, 1], [1, stride, stride, 1], 'SAME')


# In[8]:


def max_pool(input, s, stride):
    return tf.nn.max_pool(input, [1, s, s, 1], [1, stride, stride, 1], 'SAME')


# In[9]:


def loss(logits, targets):
    # Get rid of extra dimensions and cast targets into integers
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Calculate cross entropy from logits and targets
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    # Take the average loss across batch size
    #cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy') + tf.add_n(tf.get_collection('losses'))
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


# In[10]:


# Train step
def train(loss_value, model_learning_rate):
    # Create optimizer
    # my_optimizer = tf.train.MomentumOptimizer(model_learning_rate, momentum=0.9)

    my_optimizer = tf.train.AdamOptimizer(model_learning_rate)
    # Initialize train step
    train_step = my_optimizer.minimize(loss_value)
    return train_step


# In[11]:


# Accuracy function
def accuracy_of_batch(logits, targets):
    # Make sure targets are integers and drop extra dimensions
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Get predicted values by finding which logit is the greatest
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    # Check if they are equal across the batch
    predicted_correctly = tf.equal(batch_predictions, targets)
    # Average the 1's and 0's (True's and False's) across the batch size
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy


# In[12]:


def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding="bytes").item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    get_var = tf.get_variable(subkey).assign(data)
                    session.run(get_var)


# In[13]:


def fc(x, num_in, num_out, name):
    with tf.variable_scope(name) as scope:
        Wfc = weight_variable([num_in, num_out], name=name)
        bfc = bias_variable([num_out], name=name)

        tf.summary.histogram(name + "/weights", Wfc)
        tf.summary.histogram(name + "/biases", bfc)

        act = tf.nn.xw_plus_b(x, Wfc, bfc, name=name + '/op')

        return act

### building block & transition block: for ResNet18, ResNet34
def BdBlock(input, in_feature_dim, out_feature_dim, name, downsample=False, is_training=True):
    with tf.variable_scope(name) as scope:            
        ### test1 BN=0.9  test=0.99       
        current=input
        ## batch norm
        if is_training:
            tmp = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                                   updates_collections=tf.GraphKeys.UPDATE_OPS, scope=name+ '/bn_1')
        else:
            tmp = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training, scope=name+ '/bn_1')        
        ## relu
        tmp = tf.nn.relu(tmp)        
        ## 3*3 conv
        if downsample:
            current=conv2d(current,in_feature_dim,out_feature_dim,1,2,is_training, 
                                   name=name + '/conv_layer_init')
            tmp=conv2d(tmp,in_feature_dim,out_feature_dim,3,2,is_training, 
                                   name=name + '/conv_layer_1')
        else:
            tmp=conv2d(tmp,in_feature_dim,out_feature_dim,3,1,is_training, 
                                   name=name + '/conv_layer_1')
            
        ## batch norm
        if is_training:
            tmp = tf.contrib.layers.batch_norm(tmp, decay=0.99, scale=True, is_training=is_training,
                                                   updates_collections=tf.GraphKeys.UPDATE_OPS, scope=name+ '/bn_2')
        else:
            tmp = tf.contrib.layers.batch_norm(tmp, decay=0.99, scale=True, is_training=is_training, scope=name+ '/bn_2')        
        # relu
        tmp = tf.nn.relu(tmp)        
        ## 3*3 conv
        tmp=conv2d(tmp,out_feature_dim,out_feature_dim,3,1,is_training, 
                                   name=name + '/conv3_layer_2')
        ## sum        
        return current+tmp
 
### Res 18
def ResNet18(xs,is_training):
    current = tf.reshape(xs, [-1, 224, 224, 3])
    ## conv1   
    current = conv2d(current, 3, 64, 7, 2, name='conv1')
    current = max_pool(current, 3, 2)    
    
    ## conv2   3 bdblock
    for id in range(2):
        current=BdBlock(current, 64, 64, downsample=False, is_training=is_training, name='conv2_%d'%id)
    
    ## conv3 4 bdblock
    current=BdBlock(current, 64, 128, downsample=True, is_training=is_training, name='conv3_trans')
    for id in range(1):
        current=BdBlock(current, 128, 128, downsample=False, is_training=is_training, name='conv3_%d'%id)

    ## conv4 6 bdblock
    current=BdBlock(current, 128, 256, downsample=True, is_training=is_training, name='conv4_trans')
    for id in range(1):
        current=BdBlock(current, 256, 256, downsample=False, is_training=is_training, name='conv4_%d'%id)
    
    ## conv5 3 bdblock
    current=BdBlock(current, 256, 512, downsample=True, is_training=is_training, name='conv5_trans')
    for id in range(1):
        current=BdBlock(current, 512, 512, downsample=False, is_training=is_training, name='conv5_%d'%id)
    
    current = avg_pool(current, 7, 7)
    
    final_dim=512

    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 21, name='fc')
    
    return output
    


### Res 34
    
def ResNet34(xs,is_training):
    current = tf.reshape(xs, [-1, 224, 224, 3])
    ## conv1   
    current = conv2d(current, 3, 64, 7, 2, name='conv1')
    current = max_pool(current, 3, 2)    
    
    ## conv2   3 bdblock
    for id in range(3):
        current=BdBlock(current, 64, 64, downsample=False, is_training=is_training, name='conv2_%d'%id)
    
    ## conv3 4 bdblock
    current=BdBlock(current, 64, 128, downsample=True, is_training=is_training, name='conv3_trans')
    for id in range(3):
        current=BdBlock(current, 128, 128, downsample=False, is_training=is_training, name='conv3_%d'%id)

    ## conv4 6 bdblock
    current=BdBlock(current, 128, 256, downsample=True, is_training=is_training, name='conv4_trans')
    for id in range(5):
        current=BdBlock(current, 256, 256, downsample=False, is_training=is_training, name='conv4_%d'%id)
    
    ## conv5 3 bdblock
    current=BdBlock(current, 256, 512, downsample=True, is_training=is_training, name='conv5_trans')
    for id in range(2):
        current=BdBlock(current, 512, 512, downsample=False, is_training=is_training, name='conv5_%d'%id)
    
    current = avg_pool(current, 7, 7)
    
    final_dim=512

    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 21, name='fc')
    
    return output

### bottleneck building block: for ResNet50, ResNet101, ResNet152
    ### channels=indim, channel*4=output, outdim=4*indim
def BotBdBlock(input, in_feature_dim, out_feature_dim, name, downsample=False, is_training=True):
    with tf.variable_scope(name) as scope:            
        ### test1 BN=0.9  test=0.99       
        current=input
        ## batch norm
        if is_training:
            tmp = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                                   updates_collections=tf.GraphKeys.UPDATE_OPS, scope=name+ '/bn_1')
        else:
            tmp = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training, scope=name+ '/bn_1')        
        ## relu
        shortcut = tf.nn.relu(tmp)        
        
        channel=shortcut.shape[3]
        
        # 1*1 conv
        tmp=conv2d(shortcut,channel,in_feature_dim,1,1,is_training, 
                                   name=name + '/conv_layer_1_1')
        ## batch norm
        if is_training:
            tmp = tf.contrib.layers.batch_norm(tmp, decay=0.99, scale=True, is_training=is_training,
                                                   updates_collections=tf.GraphKeys.UPDATE_OPS, scope=name+ '/bn_2')
        else:
            tmp = tf.contrib.layers.batch_norm(tmp, decay=0.99, scale=True, is_training=is_training, scope=name+ '/bn_2')        
        ## relu
        tmp = tf.nn.relu(tmp)
               
        ## 3*3 conv
        if downsample:
            shortcut=conv2d(shortcut,channel,out_feature_dim,1,2,is_training, 
                                   name=name + '/conv_layer_init')
            tmp=conv2d(tmp,in_feature_dim,in_feature_dim,3,2,is_training, 
                                   name=name + '/conv_layer_1')
        else:
            shortcut=conv2d(shortcut,channel,out_feature_dim,1,1,is_training, 
                                   name=name + '/conv_layer_init')
            tmp=conv2d(tmp,in_feature_dim,in_feature_dim,3,1,is_training, 
                                   name=name + '/conv_layer_1')
        
        ## batch norm
        if is_training:
            tmp = tf.contrib.layers.batch_norm(tmp, decay=0.99, scale=True, is_training=is_training,
                                                   updates_collections=tf.GraphKeys.UPDATE_OPS, scope=name+ '/bn_3')
        else:
            tmp = tf.contrib.layers.batch_norm(tmp, decay=0.99, scale=True, is_training=is_training, scope=name+ '/bn_3')        
        # relu
        tmp = tf.nn.relu(tmp)        
        ## 1*1 conv
        tmp=conv2d(tmp,in_feature_dim,out_feature_dim,1,1,is_training, 
                                   name=name + '/conv_layer_2')
        
        ## sum        
        return tmp+shortcut
    
    ##https://github.com/taki0112/ResNet-Tensorflow/blob/master/ops.py

### Res50
def ResNet50(xs,is_training):
    current = tf.reshape(xs, [-1, 224, 224, 3])
    ## conv1   
    current = conv2d(current, 3, 64, 7, 2, name='conv1')
    current = max_pool(current, 3, 2)    
    
    ## conv2   3 bdblock    
    for id in range(3):
        current=BotBdBlock(current, 64, 256, downsample=False, is_training=is_training, name='conv2_%d'%id)
    
    ## conv3 4 bdblock
    current=BotBdBlock(current, 128, 512, downsample=True, is_training=is_training, name='conv3_trans')
    for id in range(3):
        current=BotBdBlock(current, 128, 512, downsample=False, is_training=is_training, name='conv3_%d'%id)

    ## conv4 6 bdblock
    current=BotBdBlock(current, 256, 1024, downsample=True, is_training=is_training, name='conv4_trans')
    for id in range(5):
        current=BotBdBlock(current, 256, 1024, downsample=False, is_training=is_training, name='conv4_%d'%id)
    
    ## conv5 3 bdblock
    current=BotBdBlock(current, 512, 2048, downsample=True, is_training=is_training, name='conv5_trans')
    for id in range(2):
        current=BotBdBlock(current, 512, 2048, downsample=False, is_training=is_training, name='conv5_%d'%id)
    
    current = avg_pool(current, 7, 7)
    
    final_dim=2048

    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 21, name='fc')
    
    return output

## Res101
def ResNet101(xs,is_training):
    current = tf.reshape(xs, [-1, 224, 224, 3])
    ## conv1   
    current = conv2d(current, 3, 64, 7, 2, name='conv1')
    current = max_pool(current, 3, 2)    
    
    ## conv2   3 bdblock    
    for id in range(3):
        current=BotBdBlock(current, 64, 256, downsample=False, is_training=is_training, name='conv2_%d'%id)
    
    ## conv3 4 bdblock
    current=BotBdBlock(current, 128, 512, downsample=True, is_training=is_training, name='conv3_trans')
    for id in range(3):
        current=BotBdBlock(current, 128, 512, downsample=False, is_training=is_training, name='conv3_%d'%id)

    ## conv4 23 bdblock
    current=BotBdBlock(current, 256, 1024, downsample=True, is_training=is_training, name='conv4_trans')
    for id in range(22):
        current=BotBdBlock(current, 256, 1024, downsample=False, is_training=is_training, name='conv4_%d'%id)
    
    ## conv5 3 bdblock
    current=BotBdBlock(current, 512, 2048, downsample=True, is_training=is_training, name='conv5_trans')
    for id in range(2):
        current=BotBdBlock(current, 512, 2048, downsample=False, is_training=is_training, name='conv5_%d'%id)
    
    current = avg_pool(current, 7, 7)
    
    final_dim=2048

    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 21, name='fc')
    
    return output

## Res152
def ResNet152(xs,is_training):
    current = tf.reshape(xs, [-1, 224, 224, 3])
    ## conv1   
    current = conv2d(current, 3, 64, 7, 2, name='conv1')
    current = max_pool(current, 3, 2)    
    
    ## conv2   3 bdblock    
    for id in range(3):
        current=BotBdBlock(current, 64, 256, downsample=False, is_training=is_training, name='conv2_%d'%id)
    
    ## conv3 8 bdblock
    current=BotBdBlock(current, 128, 512, downsample=True, is_training=is_training, name='conv3_trans')
    for id in range(7):
        current=BotBdBlock(current, 128, 512, downsample=False, is_training=is_training, name='conv3_%d'%id)

    ## conv4 36 bdblock
    current=BotBdBlock(current, 256, 1024, downsample=True, is_training=is_training, name='conv4_trans')
    for id in range(35):
        current=BotBdBlock(current, 256, 1024, downsample=False, is_training=is_training, name='conv4_%d'%id)
    
    ## conv5 3 bdblock
    current=BotBdBlock(current, 512, 2048, downsample=True, is_training=is_training, name='conv5_trans')
    for id in range(2):
        current=BotBdBlock(current, 512, 2048, downsample=False, is_training=is_training, name='conv5_%d'%id)
    
    current = avg_pool(current, 7, 7)
    
    final_dim=2048

    current = tf.reshape(current, [-1, final_dim])
    output = fc(current, final_dim, 21, name='fc')
    
    return output





# In[18]:

def main():
    # Dataset path
    train_tfrecords = 'train.tfrecords'
    test_tfrecords = 'test.tfrecords'

    # Learning params  ImageNet  LR=0.001
    learning_rate = 0.001
    training_iters = 50400  # 1 epoch, 1680 iter
    batch_size = 20

    # Load batch
    train_img, train_label = input_pipeline(train_tfrecords, batch_size)
    test_img, test_label = input_pipeline(test_tfrecords, batch_size)

    # Model
    with tf.variable_scope('model_definition') as scope:
        train_output = ResNet50(train_img, is_training=True)
        scope.reuse_variables()
        test_output = ResNet50(test_img, is_training=False)
        
    # Loss and optimizer
    loss_op = loss(train_output, train_label)
    #l2_regularization = l2_reg()
    #loss_sum = loss_op + 0.0005 * l2_regularization

    tf.summary.scalar('loss', loss_op)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = train(loss_op, learning_rate)
        test_loss_op = loss(test_output, test_label)
        # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

    # Evaluation
    train_accuracy = accuracy_of_batch(train_output, train_label)
    tf.summary.scalar("train_accuracy", train_accuracy)

    test_accuracy = accuracy_of_batch(test_output, test_label)
    tf.summary.scalar("test_accuracy", test_accuracy)

    # Init
    init = tf.global_variables_initializer()

    # Summary
    merged_summary_op = tf.summary.merge_all()

    # Create Saver
    # saver = tf.train.Saver(tf.trainable_variables())
    ### the default saver is tf.train.Saver() However use this leads to mistakes
    # saver = tf.train.Saver()

    ### new solution
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list,max_to_keep=1000)

    # Launch the graph
    with tf.Session() as sess:
        print('Init variable')
        sess.run(init)

        #load_ckpt_path = 'checkpoint/my-model.ckpt-346920'
        #saver.restore(sess, load_ckpt_path)

        summary_writer = tf.summary.FileWriter('logs', sess.graph)

        print('Start training')
        # coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess)
        for step in range(training_iters):
            step += 1
            # _, loss_value = sess.run([train_op, loss_op])
            # print('Generation {}: Loss = {:.5f}'.format(step, loss_value))
            # print(Wfc1value[1, 1], Wfc2value[1, 1])
            _, loss_value, test_loss_value = sess.run([train_op, loss_op, test_loss_op])
            print('Generation {}: Loss = {:.5f}     Test Loss={:.5f}'.format(step, loss_value, test_loss_value))

            # Display testing status
            if step % 40 == 0:
                acc1 = sess.run(train_accuracy)
                print(' --- Train Accuracy = {:.2f}%.'.format(100. * acc1))
                acc2 = sess.run(test_accuracy)
                print(' --- Test Accuracy = {:.2f}%.'.format(100. * acc2))

            if step % 40 == 0:
                summary_str = sess.run(merged_summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step % 840 == 0:
                saver.save(sess, 'checkpoint/my-model.ckpt', global_step=step)

        print("Finish Training and validation!")

        # coord.request_stop()
        # coord.join(threads)


# In[19]:


if __name__ == '__main__':
    main()



