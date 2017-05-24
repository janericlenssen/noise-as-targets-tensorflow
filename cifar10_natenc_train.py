import model
import numpy as np
import tensorflow as tf
np.random.seed(3333)
import timeit
import utils


params = dict(batch_size=256,
              model_dir='/media/jan/DataExt4/pycharm/natenc-cifar10-allclass',
              data_dir='/media/jan/DataExt4/BenchmarkDataSets/Classification/cifar-10-batches-py',
              input_type='cifar10',
              use_grayscale=False,
              use_gradient_images=True,
              lr=0.0001,
              mlp_lr=0.001,
              lr_update_step=15000,
              decay_steps=2,
              num_epochs=20000,
              output_every=100,
              train_mlp_every=10,
              mlp_epochs=200,
              z_dim=16,
              num_classes=10)


# Load cifar
data_train, labels_train, data_test, labels_test = \
    utils.load_cifar_XandY(params['data_dir'])

# Train data
data_train_prep = data_train
data_train_mlp = data_train

# Test data
data_test_prep = data_test

# Target reps
targetReps = utils.generateTargetReps(data_train.shape[0], params['z_dim'])


# setup gan
nat_enc = model.NATEnc(params)
# tf.global_variables_initializer().run()

# MLP reset op
mlp_reset_op = tf.variables_initializer(nat_enc.mlp_vars)

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(params['model_dir'])
sv = tf.train.Supervisor(logdir=params['model_dir'],
                        is_chief=True,
                        saver=saver,
                        summary_op=None,
                        summary_writer=summary_writer,
                        save_model_secs=300,
                        global_step=nat_enc.step,
                        ready_for_local_init_op=None)

gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(allow_soft_placement=True,
                            gpu_options=gpu_options)
lr = params['lr']
# set up the TF session and init all ops and variables
with sv.prepare_or_wait_for_session(config=sess_config) as sess:

    # Pick a big sample from z and project it through G and compare to pdf_x (original data pdf)
    # this is not data to be trained on, but to check G projections
    #sample_z = params['z_prior'](-1, 1, [real_data_train.shape[0], params['z_dim']]).astype(np.float32)
    batches_per_epoch = data_train_prep.shape[0] / params['batch_size']
    counter = 0
    curr_epoch = -1
    batch_timings = []
    decay_steps = 0
    for counter in xrange(params['num_epochs'] * batches_per_epoch):
        if counter % batches_per_epoch == 0:
            # New epoch
            batch_idx = 0
            curr_epoch += 1
            # Randomize order
            targetReps, data_train_prep = utils.shuffle_together(targetReps, data_train_prep)

        beg_t = timeit.default_timer()
        # sample a batch from prior pdf z
        # get a batch of samples from gtruth pdf
        batch_x_real = data_train_prep[batch_idx:(batch_idx + params['batch_size'])]
        batch_target = targetReps[batch_idx:(batch_idx + params['batch_size'])]
        if (curr_epoch +1) % 3 == 0:
            # Get Current Representations
            feed_dict = {nat_enc.input_train_ph: batch_x_real,nat_enc.dropout_keep_prob:1.0}
            fetch_dict = {
                    "reps": nat_enc.representation
                }
            result = sess.run(fetch_dict,feed_dict=feed_dict)

            # Optimize Assignment with Hungarian Algorithm
            batch_target = utils.calc_optimal_target_permutation(result['reps'], batch_target)
            targetReps[batch_idx:(batch_idx + params['batch_size'])] = batch_target

        # Optimize Network Weights
        feed_dict = {nat_enc.targets: batch_target, nat_enc.input_train_ph: batch_x_real,nat_enc.dropout_keep_prob:0.5}

        fetch_dict = {
                "train": nat_enc.train_op
            }
        if counter % params['output_every'] == 0 and counter!= 0:
            fetch_dict.update({
                "summary": nat_enc.summary_op,
                "loss": nat_enc.loss
            })
        result = sess.run(fetch_dict,feed_dict=feed_dict)


        end_t = timeit.default_timer()
        batch_timings.append(end_t - beg_t)
        if counter % params['output_every'] == 0 and counter!= 0:
            loss = result['loss']
            print("Epoch {}, Step [{}/{}] Loss: {:.6f}". \
                      format(curr_epoch, counter%batches_per_epoch, batches_per_epoch, loss))


            summary_writer.add_summary(result['summary'], counter)
            summary_writer.flush()

        if counter % params['lr_update_step'] == params['lr_update_step'] - 1 and decay_steps<params['decay_steps']:
            lr = lr*0.5
            decay_steps+=1
            sess.run([nat_enc.lr_update],feed_dict={nat_enc.new_lr: lr})

        # Train mlp
        if curr_epoch % params['train_mlp_every'] == 0 and counter % batches_per_epoch == 0:
            # Reset MLP
            sess.run(mlp_reset_op)
            # Compute Representations
            batches_per_epoch = data_train_mlp.shape[0] / params['batch_size']
            mlp_batch_idx = 0
            # randomize order
            labels_train, data_train_mlp = utils.shuffle_together(labels_train, data_train_mlp)

            x_train_mlp = data_train_mlp[:batches_per_epoch*params['batch_size']]
            y_train = labels_train[:batches_per_epoch*params['batch_size']]

            computed_reps = []
            for mlp_step in range(batches_per_epoch):
                batch_x_real = x_train_mlp[mlp_step*params['batch_size']:(mlp_step+1)*params['batch_size']]
                reps = sess.run(nat_enc.representation,
                         feed_dict={nat_enc.input_train_ph:batch_x_real,nat_enc.dropout_keep_prob:1.0})
                computed_reps.append(reps)

            computed_reps = np.concatenate(computed_reps,axis=0)
            for mlp_step in range(params['mlp_epochs']*batches_per_epoch):
                if counter % batches_per_epoch == 0:
                    # epoch change. First time this if is true, so also init variables.
                    mlp_batch_idx = 0
                    y_train, computed_reps = utils.shuffle_together(y_train, computed_reps)

                batch_reps = computed_reps[mlp_batch_idx:(mlp_batch_idx + params['batch_size'])]
                batch_label = y_train[mlp_batch_idx:(mlp_batch_idx + params['batch_size'])]
                _, loss, top_k, step = sess.run([nat_enc.mlp_train_op,nat_enc.mlp_loss, nat_enc.mlp_top_k_from_ph, nat_enc.mlp_step],
                         feed_dict={nat_enc.mlp_labels:batch_label, nat_enc.representation_ph:batch_reps})
                top_k = np.sum(top_k)

                if mlp_step % batches_per_epoch == batches_per_epoch -1:
                    print("MLP Training Epoch {} finished, Loss: {:.6f}, Accuracy: {:.6f}". \
                              format(mlp_step/batches_per_epoch, loss, float(top_k)/params['batch_size']))
                    tag = 'mlp_train_epoch'+str(curr_epoch)
                    train_summary = tf.Summary(value=[tf.Summary.Value(tag=tag+"/loss", simple_value=loss),
                                             tf.Summary.Value(tag=tag+"/train_accuracy", simple_value=float(top_k)/params['batch_size']),
                                       ])
                    summary_writer.add_summary(train_summary,step)
                    summary_writer.flush()

                mlp_batch_idx += params['batch_size']

            # Test trained MLP
            labels_test, data_test_prep = utils.shuffle_together(labels_test, data_test_prep)
            batches = data_test_prep.shape[0] / params['batch_size']
            correct_pred = 0
            mlp_batch_idx = 0
            for _ in range(batches):
                batch_x_test = data_test_prep[mlp_batch_idx:(mlp_batch_idx + params['batch_size'])]
                batch_label = labels_test[mlp_batch_idx:(mlp_batch_idx + params['batch_size'])]
                top_k = sess.run(nat_enc.mlp_top_k,feed_dict={nat_enc.input_test_ph:batch_x_test, nat_enc.mlp_labels:batch_label,nat_enc.dropout_keep_prob:1.0})
                top_k = np.sum(top_k)
                correct_pred += top_k
                mlp_batch_idx += params['batch_size']

            accuracy = float(correct_pred)/mlp_batch_idx
            print("MLP test accuracy: {:.6f}".format(accuracy))
            test_summary = tf.Summary(value=[tf.Summary.Value(tag="mlp_test/test_accuracy", simple_value=accuracy)])
            summary_writer.add_summary(test_summary,counter)
            summary_writer.flush()


        counter += 1
        batch_idx += params['batch_size']

    print("Done training for {} epochs! Elapsed time: {}s".format(params['num_epochs'], np.sum(batch_timings)))
    print("Total amount of iterations done: ", counter)

