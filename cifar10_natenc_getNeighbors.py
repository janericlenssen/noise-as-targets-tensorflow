import model
import numpy as np
import tensorflow as tf
np.random.seed(3333)
import utils


params = dict(test_batch_size=32,
              model_dir='/media/jan/DataExt4/pycharm/natenc-cifar10-allclass',
              input_type='cifar10',
              use_grayscale=True,
              use_gradient_images=True,
              z_dim=16,
              num_classes=10)


num_tests = 1
samples = [0,10,20]
number_of_neighbors = 5

test_batch_size = params['test_batch_size']

#load kekse
data_train, labels_train, data_test, labels_test = \
    utils.load_cifar_XandY('/media/jan/DataExt4/BenchmarkDataSets/Classification/cifar-10-batches-py')

data_test_prep = data_test

# setup gan
net_enc = model.NATEnc(params)
# tf.global_variables_initializer().run()

summary_writer = tf.summary.FileWriter(params['model_dir'])
sv = tf.train.Supervisor(logdir=params['model_dir'], summary_writer=None)

gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(allow_soft_placement=True,
                            gpu_options=gpu_options)
lr = params['lr']
# set up the TF session and init all ops and variables
with sv.prepare_or_wait_for_session(config=sess_config) as sess:
    cycles = data_test_prep.shape[0]/test_batch_size
    results = []
    for i in range(cycles):
        feed_dict = {net_enc.input_test_ph: data_test_prep[i*test_batch_size:(i+1)*test_batch_size]}
        fetch_dict = {
                "reps": net_enc.representation_test
            }
        res_test = sess.run(fetch_dict,feed_dict=feed_dict)
        results.append(res_test['reps'])
    r = np.concatenate(results,axis=0)
    images = []
    for counter,sample in enumerate(samples):
        images.append(np.expand_dims(data_test[sample],axis=0))
        img_without = np.concatenate([data_test[:sample,:],data_test[sample+1:,:]],axis=0)
        r_without = np.concatenate([r[:sample,:],r[sample+1:,:]],axis=0)
        for i in range(number_of_neighbors):
            nearest_index = np.sum(np.square(r_without-r[sample]),axis=1).argmin()
            images.append(np.expand_dims(img_without[nearest_index],axis=0))
            img_without = np.concatenate([img_without[:nearest_index,:],img_without[nearest_index+1:,:]],axis=0)
            r_without = np.concatenate([r_without[:nearest_index,:],r_without[nearest_index+1:,:]],axis=0)

    printimages = np.concatenate(images,axis=0)
    utils.save_images(printimages,[len(samples),number_of_neighbors+1],'/home/jan/Documents/GANResults/natenc_neighbors/test_arange.png')
