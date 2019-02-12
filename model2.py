import tensorflow as tf

def get_weights(shape, name, init='glorot_uniform'):
    """
    Get weights and initialize them
    for the each layer.
    """
    if init == 'glorot_uniform':
        i = tf.glorot_uniform_initializer(seed=42)
    elif init == 'glorot_normal':
        i = tf.glorot_normal_initializer(seed=42)
    elif init == 'xavier_uniform':
        i = tf.contrib.layers.xavier_initializer(seed=42)
    elif init == 'xavier_normal':
        i = tf.contrib.layers.xavier_initializer(seed=42, uniform=False)
    elif init == 'he_uniform':
        i = tf.keras.initializers.he_uniform(seed=42)
    elif init == 'he_normal':
        i = tf.keras.initializers.he_normal(seed=42)
    elif init == 'raw':
        if len(shape) > 1:
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        else:
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
#         i = tf.initializers.random_normal(seed=42)

    return tf.get_variable(
        name=name,
        shape=shape,
        dtype=tf.float32,
        initializer=i,)

def flatten(X, size):
    return tf.reshape(X, [-1, size])
    

def Dense(X, size, init, name, activation):
    w = get_weights(shape=size, name='W_' + name, init=init)
    b = get_weights(shape=[size[-1]], name='b_' + name, init=init)
    
    dense = tf.matmul(X, w) + b
    print(name, size, size[-1])
    ## Applying activation

    if activation == 'relu':
        h_fc = tf.nn.relu(dense)
    elif activation == 'sigmoid':
        h_fc = tf.nn.sigmoid(dense)
    elif activation == 'leaky_relu':
        h_fc = tf.nn.leaky_relu(dense)
    elif activation == 'tanh':
        h_fc = tf.nn.tanh(dense)
    elif activation == 'atan':
        h_fc = tf.atan(dense)
    
#     if dropout >= 0.0 and dropout < 1.0:
#         return tf.nn.dropout(h_fc, keep_prob=dropout)
    return h_fc

def Conv2d(X, size, stride, init, name, padding, activation):
    """
    Get a conv layer on X for weight W and bias b
    with stride and padding
    """
    print(name, size, size[-1])
    w = get_weights(shape=size, name='W_' + name, init=init)
    b = get_weights(shape=[size[-1]], name='b_' + name, init=init)
    
    conv = tf.nn.conv2d(X, w, strides=[1, stride, stride, 1], 
                        padding=padding) + b
    
    ## Applying activation

    if activation == 'relu':
        h_conv = tf.nn.relu(conv)
    elif activation == 'sigmoid':
        h_conv = tf.nn.sigmoid(conv)
    elif activation == 'leaky_relu':
        h_conv = tf.nn.leaky_relu(conv)
    
    return h_conv


x = tf.placeholder(name='input', dtype=tf.float32, shape=[None, 66, 200, 3])
y_ = tf.placeholder(name='output', dtype=tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)

x_image = x

# first convolutional layer
h_conv1 = Conv2d(x_image, (5, 5, 3, 24), 2, 
                 init='raw', name='conv2d_1',
                 padding='VALID', activation='relu')

# second convolutional layer
h_conv2 = Conv2d(h_conv1, (5, 5, 24, 36), 2, 
                 init='raw', name='conv2d_2',
                 padding='VALID', activation='relu')

# third convolutional layer
h_conv3 = Conv2d(h_conv2, (5, 5, 36, 48), 2, 
                 init='raw', name='conv2d_3',
                 padding='VALID', activation='relu')

# fourth convolutional layer
h_conv4 = Conv2d(h_conv3, (3, 3, 48, 64), 1, 
                 init='raw', name='conv2d_4',
                 padding='VALID', activation='relu')

# fifth convolutional layer
h_conv5 = Conv2d(h_conv4, (3, 3, 64, 64), 1, 
                 init='raw', name='conv2d_5',
                 padding='VALID', activation='relu')

# Flatten layer
h_conv5_flatten = flatten(h_conv5, size=1152)

# Dense layer 1
h_fc1 = Dense(h_conv5_flatten, (1152, 1164), name='dense1',
              init='raw',
             activation='relu')
# Dropout 1
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Dense Layer 2
h_fc2 = Dense(h_fc1_drop, (1164, 100), name='dense2',
              init='raw',
             activation='relu')
# Dropout 2
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# Dense Layer 3
h_fc3 = Dense(h_fc2_drop, (100, 50), name='dense3',
              init='raw',
             activation='relu')
# Dropout 3
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

# Dense Layer 4
h_fc4 = Dense(h_fc3_drop, (50, 10), name='dense4',
              init='raw',
             activation='relu')

# Dropout 4
h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

# Output
W_fc5 = get_weights([10, 1], name='W_fc5', init='raw')
b_fc5 = get_weights([1], name='b_fc5', init='raw')
y = tf.matmul(h_fc4_drop, W_fc5) + b_fc5