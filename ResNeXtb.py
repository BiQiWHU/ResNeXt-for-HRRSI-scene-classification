# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 23:44:57 2019

@author: Qi Bi
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


def fcnobias(x, num_in, num_out, name):
    with tf.variable_scope(name) as scope:
        Wfc = weight_variable([num_in, num_out], name=name)
        #bfc = bias_variable([num_out], name=name)
        tf.summary.histogram(name + "/weights", Wfc)
        #tf.summary.histogram(name + "/biases", bfc)

        #act = tf.nn.xw_plus_b(x, Wfc, bfc, name=name + '/op')
        act=tf.matmul(x, Wfc,name=name + '/nobias')
        return act


### ResBlock

cardinality = 32 # how many split ?

def transform_layer(input,out_feature_dim,stride,is_training,name):
    channel=input.shape[3]
    x=input            
    # 1*1 conv
    x=conv2d(x,channel,out_feature_dim,1,1,is_training, name=name + '/transform_conv_1_1')
    # batch norm
    if is_training:
        x = tf.contrib.layers.batch_norm(x, decay=0.99, scale=True, is_training=is_training,
                                                   updates_collections=tf.GraphKeys.UPDATE_OPS, scope=name+ '/bn_1')
    else:
        x = tf.contrib.layers.batch_norm(x, decay=0.99, scale=True, is_training=is_training, scope=name+ '/bn_1')        
    ## relu
    x = tf.nn.relu(x)        

    ## 3*3 conv
    x=conv2d(x,out_feature_dim,out_feature_dim,3,stride,is_training, name=name + '/conv_layer_3_3')
    # batch norm
    if is_training:
        x = tf.contrib.layers.batch_norm(x, decay=0.99, scale=True, is_training=is_training,
                                                   updates_collections=tf.GraphKeys.UPDATE_OPS, scope=name+ '/bn_2')
    else:
        x = tf.contrib.layers.batch_norm(x, decay=0.99, scale=True, is_training=is_training, scope=name+ '/bn_2')        
    ## relu
    x = tf.nn.relu(x)
    return x      
    
def transition_layer(input,out_feature_dim, is_training,name):
    channel=input.shape[3]
    x=input 
    ## 1*1 conv 
    x=conv2d(x,channel,out_feature_dim,1,1,is_training, name=name + '/transit_conv_layer_1_1')
    ## batch norm
    if is_training:
        x = tf.contrib.layers.batch_norm(x, decay=0.99, scale=True, is_training=is_training,
                                                   updates_collections=tf.GraphKeys.UPDATE_OPS, scope=name+ '/bn_1')
    else:
        x = tf.contrib.layers.batch_norm(x, decay=0.99, scale=True, is_training=is_training, scope=name+ '/bn_1')        
    return x
    
def split_layer(input,out_feature_dim, cardinality, stride, is_training, name):
    x=input
    
    layers_split = list()
    
    dim=out_feature_dim//cardinality
    
    for i in range(cardinality) :
        splits = transform_layer(x, dim, stride, is_training, name=name + '_splitN_' + str(i))
        layers_split.append(splits)
    
    return tf.concat(layers_split, axis=3)
    

### ResNeXt50  
def ResBlockfirstlayer(input, out_feature_dim, cardinality, name, downsample=True,  is_training=True):
    input_dim=input.shape[3]
    
    if input_dim==out_feature_dim:
        channel=input_dim//2
        x=tf.pad(input, [[0, 0], [0, 0], [0, 0],[channel, channel]])
    else:
        x=input
        
    if downsample:
        stride=2
     
        current=split_layer(input,out_feature_dim,cardinality,stride,is_training, name=name+'split_layer')
        current=transition_layer(current,2*out_feature_dim,is_training, name=name+'trans_layer')
        
        pad_x=avg_pool(x,2,2)
      
    else:
        stride=1
        
        current=split_layer(input,out_feature_dim,cardinality,stride,is_training, name=name+'split_layer')
        current=transition_layer(current,2*out_feature_dim,is_training,name=name+'trans_layer')

        pad_x=x
        
    x=current+pad_x
    return x


def ResBlocklayer(input, out_feature_dim, cardinality, name, downsample=False,  is_training=True):
    #input_dim=input.shape[3]
    
    x=input
        
    if downsample:
        stride=2
        #channel=input_dim//2
        
        current=split_layer(x,out_feature_dim,cardinality,stride,is_training, name=name+'split_layer')
        current=transition_layer(current,2*out_feature_dim,is_training, name=name+'trans_layer')
     
        pad_x=avg_pool(x,2,2)
       
    else:
        stride=1
        
        current=split_layer(x,out_feature_dim,cardinality,stride,is_training, name=name+'split_layer')
        current=transition_layer(current,2*out_feature_dim,is_training,name=name+'trans_layer')
        #current=squeeze_excitation_layer(current,2*out_feature_dim, reduction_ratio, is_training, name=name+'squeeze_layer')
    
        pad_x=x
        
    x=current+pad_x
    return x

### ResNeXt50  nosqueeze
def ResNeXt50(xs,is_training):
    current = tf.reshape(xs, [-1, 224, 224, 3])
    ## conv1   112*112 
    current = conv2d(current, 3, 64, 7, 2, name='conv1')
    # batch norm
    if is_training:
        current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training,
                                                   updates_collections=tf.GraphKeys.UPDATE_OPS, scope='conv1/bn_1')
    else:
        current = tf.contrib.layers.batch_norm(current, decay=0.99, scale=True, is_training=is_training, scope='conv1/bn_1')        
    # relu
    current = tf.nn.relu(current)
    
    ## conv2  56*56   128,128,256,C=32, 3
    current = max_pool(current, 3, 2)
    ### padding to 128-d
    current=tf.pad(current, [[0, 0], [0, 0], [0, 0],[32, 32]])

    current=ResBlockfirstlayer(current, 128,32,name='conv2_trans',downsample=False,is_training=is_training)
    for id in range(2):
        current=ResBlocklayer(current, 128,32,name='conv2_%d'%id,downsample=False,is_training=is_training)
        
    ## conv3 28*28  256,256,512, C=32, 4
    current=ResBlockfirstlayer(current, 256,32,name='conv3_trans',downsample=True,is_training=is_training)
    for id in range(3):
        current=ResBlocklayer(current, 256,32,name='conv3_%d'%id,downsample=False,is_training=is_training)
    
    ## conv4  14*14   512,512,1024   C=32, 6
    current=ResBlockfirstlayer(current, 512,32,name='conv4_trans',downsample=True,is_training=is_training)
    for id in range(5):
        current=ResBlocklayer(current, 512,32,name='conv4_%d'%id,downsample=False,is_training=is_training)
        
    ## conv5  7*7   1024,1024,2048,  C=32, 3
    current=ResBlockfirstlayer(current, 1024,32,name='conv5_trans',downsample=True,is_training=is_training)
    for id in range(2):
        current=ResBlocklayer(current, 1024,32,name='conv5_%d'%id,downsample=False,is_training=is_training)
    
    ## gloabl avearge pooling
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

    # Learning params  原来imagenet的学习率是0.001
    learning_rate = 0.001
    training_iters = 50400  # 一个epoch1680次
    batch_size = 20

    # Load batch
    train_img, train_label = input_pipeline(train_tfrecords, batch_size)
    test_img, test_label = input_pipeline(test_tfrecords, batch_size)

    # Model
    with tf.variable_scope('model_definition') as scope:
        train_output = ResNeXt50(train_img, is_training=True)
        scope.reuse_variables()
        test_output = ResNeXt50(test_img, is_training=False)
        

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



