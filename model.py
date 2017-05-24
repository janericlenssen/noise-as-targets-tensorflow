import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


class NATEnc:
    def __init__(self, params):
        self.params = params
        # Placeholders for Input
        self.input_train_ph = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        self.input_test_ph = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
        self.targets = tf.placeholder(dtype=tf.float32, shape=[None, self.params['z_dim']])
        self.representation_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.params['z_dim']])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.new_lr = tf.placeholder(tf.float32)
        self.lr = tf.Variable(params['lr'])
        self.k = tf.Variable(0., trainable=False, name='k')
        self.step = tf.Variable(0, name='step', trainable=False)

        # MLP things
        self.mlp_labels = tf.placeholder(dtype=tf.int32, shape=[None])
        self.new_mlp_lr = tf.placeholder(tf.float32)
        self.mlp_lr = tf.Variable(params['mlp_lr'])
        self.mlp_step = tf.Variable(0, name='mlp_step', trainable=False)

        # Input Preprocessing
        self.channels = 1
        self.input_train, self.input_test = self.image_augmentation(self.input_train_ph, self.input_test_ph)

        # Model Def
        self.representation, self.enc_vars = self.encoder(self.input_train)
        self.representation_test, self.enc_vars = self.encoder(self.input_test,reuse=True)

        # MLP Classifier
        self.logits_from_ph, _ = self.mlp_classifier(self.representation_ph)
        self.mlp_top_k_from_ph = tf.nn.in_top_k(self.logits_from_ph,self.mlp_labels,1)

        self.logits, self.mlp_vars = self.mlp_classifier(self.representation_test,reuse=True)
        self.mlp_top_k = tf.nn.in_top_k(self.logits,self.mlp_labels,1)

        # Losses
        self.loss = self.calc_loss()
        self.mlp_loss = self.calc_mlp_loss()

        # Train Ops
        self.train_op = self.train()
        self.mlp_train_op = self.train_mlp()

        # Summary Op
        self.summary_op = self.summaries()
        self.mlp_summary_op = self.mlp_summaries()

        # Update lr op
        self.lr_update = tf.assign(self.lr, self.new_lr, name='lr_update')
        self.mlp_lr_update = tf.assign(self.mlp_lr, self.new_mlp_lr, name='mlp_lr_update')

    def encoder(self, x, reuse=False, data_format='NHWC'):
        with tf.variable_scope("Encoder", reuse=reuse) as vs:
            z_dim = self.params['z_dim']
            repeat_num = 4
            hidden_num = 64

            # Encoder
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            for idx in range(repeat_num):
                channel_num = hidden_num * (idx + 1)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                if idx < repeat_num - 1:
                    x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
            dim = np.prod([x.get_shape().as_list()[1:]])
            x = tf.reshape(x, [-1, dim])
            x = tf.nn.dropout(x,self.dropout_keep_prob)
            x = slim.fully_connected(x, channel_num, activation_fn=tf.nn.elu)
            x = tf.nn.dropout(x,self.dropout_keep_prob)
            x = slim.fully_connected(x, z_dim, activation_fn=None)
            representation = tf.nn.l2_normalize(x,1)

        variables = tf.contrib.framework.get_variables(vs)

        return representation, variables

    def mlp_classifier(self, x, data_format='NHWC', reuse=False):
        with tf.variable_scope("mlp",reuse=reuse) as vs:
            z_dim = self.params['z_dim']
            x = slim.fully_connected(x, self.params['num_classes']*20, activation_fn=tf.nn.relu)
            x = slim.fully_connected(x, self.params['num_classes']*20, activation_fn=tf.nn.relu)
            x = slim.fully_connected(x, self.params['num_classes'], activation_fn=None)

        variables = tf.contrib.framework.get_variables(vs)
        return x, variables

    def calc_loss(self):
        return tf.reduce_mean(self.squared_l2(self.targets,self.representation))

    def calc_mlp_loss(self):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits_from_ph,self.mlp_labels))

    def squared_l2(self, data_in, data_out):
        return tf.reduce_sum(tf.square(data_in - data_out),axis=1)

    def l1(self, data_in, data_out):
        return tf.reduce_sum(tf.abs(data_in - data_out),axis=1)

    def train(self):
        d_optimizer = tf.train.AdamOptimizer(self.lr)
        d_train_op = d_optimizer.minimize(self.loss, var_list=self.enc_vars, global_step=self.step)
        #    d_train_op = tf.no_op()
        return d_train_op

    def train_mlp(self):
        d_optimizer = tf.train.AdamOptimizer(self.mlp_lr)
        d_train_op = d_optimizer.minimize(self.mlp_loss, var_list=self.mlp_vars, global_step=self.mlp_step)
        #    d_train_op = tf.no_op()
        return d_train_op


    def summaries(self):
        summaries = [
        tf.summary.scalar("loss/d_loss", self.loss),
        tf.summary.scalar("misc/step", self.step),
        tf.summary.scalar("misc/lr", self.lr),
        tf.summary.image("training/input",self.input_train)
                ]
        return tf.summary.merge(summaries)

    def mlp_summaries(self):
        summaries = [
        tf.summary.scalar("mlp_loss/loss", self.mlp_loss),
        tf.summary.scalar("mlp_loss/accuracy", tf.to_float(tf.reduce_sum(tf.to_int32(self.mlp_top_k)))/self.params['batch_size'])
                ]
        return tf.summary.merge(summaries)

    def image_augmentation(self, train_data, test_data):
        train_data = tf.map_fn(lambda img: tf.image.flip_left_right(img), train_data)
        train_data = tf.map_fn(lambda img: tf.image.random_brightness(img,max_delta=63), train_data)
        train_data = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0.2, upper=1.8),train_data)

        if self.params['use_grayscale']:
            train_data = tf.map_fn(lambda img: tf.image.rgb_to_grayscale(img), train_data)
        if self.params['use_gradient_images']:
            train_data = self.apply_sobel(train_data)
        # self.input_real = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.input_real)

        test_data = test_data
        if self.params['use_grayscale']:
            test_data = tf.map_fn(lambda img: tf.image.rgb_to_grayscale(img), test_data)
        if self.params['use_gradient_images']:
            test_data = self.apply_sobel(test_data)
        # self.input_test = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.input_test)

        train_data = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img,30,30),train_data)
        train_data = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img,42,42),train_data)
        if self.params['use_grayscale']:
            train_data = tf.map_fn(lambda img: tf.random_crop(img,[32,32,1]),train_data)
        else:
            train_data = tf.map_fn(lambda img: tf.random_crop(img,[32,32,3]),train_data)
        return train_data, test_data

    def apply_sobel(self, input):
        channels = tf.split(3,input.get_shape().as_list()[3],input)
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.transpose(sobel_x, [1, 0, 2, 3])
        channels_out = []
        for channel in channels:
            dx = tf.nn.conv2d(channel, sobel_x,strides=[1, 1, 1, 1], padding='SAME')
            dy = tf.nn.conv2d(channel, sobel_y,strides=[1, 1, 1, 1], padding='SAME')
            channels_out.append(tf.sqrt(tf.square(dx) + tf.square(dy)))
        return tf.concat(3,channels_out)