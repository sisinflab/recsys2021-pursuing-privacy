import itertools
import sys

import gflags
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import rankdata
from tensorflow_privacy.privacy.analysis.gdp_accountant import compute_eps_poisson
from tensorflow_privacy.privacy.analysis.gdp_accountant import compute_mu_poisson
from tensorflow_privacy.privacy.optimizers import dp_optimizer

from utils import splitting_leave_one_out

sampling_batch = 10000
microbatches = 10000
num_examples = 80419

tf.compat.v1.logging.set_verbosity(3)

#### FLAGS
FLAGS = gflags.FLAGS
gflags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
                   'train with vanilla SGD.')
gflags.DEFINE_float('learning_rate', .005, 'Learning rate for training')
gflags.DEFINE_float('noise_multiplier', 0.55,
                    'Ratio of the standard deviation to the clipping norm')
gflags.DEFINE_float('l2_norm_clip', 5, 'Clipping norm')
gflags.DEFINE_integer('epochs', 10, 'Number of epochs')
gflags.DEFINE_integer('max_mu', 3, 'GDP upper limit')
gflags.DEFINE_string('model_dir', None, 'Model directory')


def nn_model_fn(features, labels, mode):

    n_latent_factors_user = 64
    n_latent_factors_movie = 64
    n_latent_factors_mf = 64

    user_input = tf.reshape(features['user'], [-1, 1])
    item_input = tf.reshape(features['movie'], [-1, 1])

    # number of users: 610; number of movies: 9724
    mf_embedding_user = tf.keras.layers.Embedding(
        610, n_latent_factors_mf, input_length=1)
    mf_embedding_item = tf.keras.layers.Embedding(
        9724, n_latent_factors_mf, input_length=1)
    mlp_embedding_user = tf.keras.layers.Embedding(
        610, n_latent_factors_user, input_length=1)
    mlp_embedding_item = tf.keras.layers.Embedding(
        9724, n_latent_factors_movie, input_length=1)

    # GMF part
    # Flatten the embedding vector as latent features in GMF
    mf_user_latent = tf.keras.layers.Flatten()(mf_embedding_user(user_input))
    mf_item_latent = tf.keras.layers.Flatten()(mf_embedding_item(item_input))
    # Element-wise multiply
    mf_vector = tf.keras.layers.multiply([mf_user_latent, mf_item_latent])

    # MLP part
    # Flatten the embedding vector as latent features in MLP
    mlp_user_latent = tf.keras.layers.Flatten()(mlp_embedding_user(user_input))
    mlp_item_latent = tf.keras.layers.Flatten()(mlp_embedding_item(item_input))
    # Concatenation of two latent features
    mlp_vector = tf.keras.layers.concatenate([mlp_user_latent, mlp_item_latent])

    mlp_1 = tf.keras.layers.Dense(256,  activation='relu')(mlp_vector)
    mlp_2 = tf.keras.layers.Dense(128, activation='relu')(mlp_1)
    mlp_3 = tf.keras.layers.Dense(64, activation='relu')(mlp_2)

    predict_vector = tf.keras.layers.concatenate([mf_vector, mlp_3])

    logits = tf.keras.layers.Dense(1)(predict_vector)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'score': tf.nn.sigmoid(logits)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss as a vector (to support microbatches in DP-SGD).
    vector_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE, from_logits=True)(
        tf.expand_dims(labels, -1), logits)
    # Define mean of loss across minibatch (for reporting through tf.Estimator).
    scalar_loss = tf.reduce_mean(vector_loss)

    # Configure the training op (for TRAIN mode).
    if mode == tf.estimator.ModeKeys.TRAIN:
        if FLAGS.dpsgd:
            # Use DP version of GradientDescentOptimizer. Other optimizers are
            # available in dp_optimizer. Most optimizers inheriting from
            # tf.train.Optimizer should be wrappable in differentially private
            # counterparts by calling dp_optimizer.optimizer_from_args().
            optimizer = dp_optimizer.DPAdamGaussianOptimizer(
                l2_norm_clip=FLAGS.l2_norm_clip,
                noise_multiplier=FLAGS.noise_multiplier,
                num_microbatches=microbatches,
                learning_rate=FLAGS.learning_rate)
            opt_loss = vector_loss
        else:
            optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=FLAGS.learning_rate)
            opt_loss = scalar_loss

        global_step = tf.compat.v1.train.get_global_step()
        train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
        # In the following, we pass the mean of the loss (scalar_loss) rather than
        # the vector_loss because tf.estimator requires a scalar loss. This is only
        # used for evaluation and debugging by tf.estimator. The actual loss being
        # minimized is opt_loss defined above and passed to optimizer.minimize().
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=scalar_loss, train_op=train_op)


    # Add evaluation metrics (for EVAL mode).
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'rmse':
                tf.compat.v1.metrics.root_mean_squared_error(
                    labels=tf.cast(labels, tf.float32),
                    predictions=tf.tensordot(
                        a=tf.nn.sigmoid(logits),
                        b=tf.constant(np.array([1]), dtype=tf.float32),
                        axes=1))
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=scalar_loss, eval_metric_ops=eval_metric_ops)
    return None


if __name__ == "__main__":
    argv = FLAGS(sys.argv)
    data = pd.read_csv(
        'data/movielens/dataset.csv')
    data = data.astype('int')
    n_users = len(set(data['userId']))
    n_movies = len(set(data['movieId']))
    print('number of movie: ', n_movies)
    print('number of user: ', n_users)

    # give unique dense movie index to movieId
    data['movieIndex'] = rankdata(data['movieId'], method='dense')
    # minus one to reduce the minimum value to 0, which is the start of col index

    print('number of ratings:', data.shape[0])
    print('percentage of sparsity:',
          (1 - data.shape[0] / n_users / n_movies) * 100, '%')

    train, test = splitting_leave_one_out(data)

    train_data, test_data, _ = train.values - 1, test.values - 1, np.mean(train['rating'])

    # Instantiate the tf.Estimator.
    ml_classifier = tf.estimator.Estimator(model_fn=nn_model_fn, model_dir=FLAGS.model_dir)

    full_eval = pd.DataFrame(list(itertools.product(set(train_data[:, 0]), set(train_data[:, 4]))))

    train_item_filter = full_eval.groupby([0])\
        .apply(lambda x: x[1].isin(train_data[train_data[:, 0] == x[0].unique()][:, 1])).reset_index(drop=True)

    full_eval = full_eval[~train_item_filter]
    full_eval["rank_first"] = full_eval.groupby(0).rank(method='first', ascending=False, axis=1)
    full_eval["test_flag"] = full_eval.apply(
        lambda x: x["rank_first"] <= 100, axis=1)
    full_eval = full_eval[full_eval["test_flag"] == True].drop(columns=["rank_first", "test_flag"]).reset_index(drop=True)
    # Create tf.Estimator input functions for the training and test data.
    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={
            'user': full_eval[0].values,
            'movie': full_eval[1].values
        },
        num_epochs=1,
        batch_size=len(full_eval),
        shuffle=False)


    steps_per_epoch = num_examples // sampling_batch
    test_accuracy_list = []
    for epoch in range(1, FLAGS.epochs + 1):
        for _ in range(steps_per_epoch):
            whether = np.random.random_sample(num_examples) > (
                    1 - sampling_batch / num_examples)
            subsampling = [i for i in np.arange(num_examples) if whether[i]]
            microbatches = len(subsampling)

            train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
                x={
                    'user': train_data[subsampling, 0],
                    'movie': train_data[subsampling, 4]
                },
                y=(train_data[subsampling, 2] >= 4).astype(np.float32),
                batch_size=len(subsampling),
                num_epochs=1,
                shuffle=False)
            # Train the model for one step.
            ml_classifier.train(input_fn=train_input_fn, steps=1)

        # Evaluate the model and print results
        # eval_results = ml_classifier.evaluate(input_fn=eval_input_fn)
        # test_accuracy = eval_results['rmse']
        # test_accuracy_list.append(test_accuracy)
        # print('Test RMSE after %d epochs is: %.3f' % (epoch, test_accuracy))

        # Compute the privacy budget expended so far.
        if FLAGS.dpsgd:
            eps = compute_eps_poisson(epoch, FLAGS.noise_multiplier, num_examples,
                                      sampling_batch, 1e-6)
            mu = compute_mu_poisson(epoch, FLAGS.noise_multiplier, num_examples,
                                    sampling_batch)
            print('For delta=1e-6, the current epsilon is: %.2f' % eps)
            print('For delta=1e-6, the current mu is: %.2f' % mu)

            if mu > FLAGS.max_mu:
                break
        else:
            print('Trained with vanilla non-private SGD optimizer')

    score = ml_classifier.predict(input_fn=eval_input_fn, yield_single_examples=False)

    for results in score:
        full_eval[2] = results['score']
        prediction = full_eval.sort_values([0, 2], ascending=[True, False])

    precisions = []
    recalls = []

    for u in prediction[0].unique():
        top_k = prediction[prediction[0] == u][1][:10]
        n_rel_and_rec_k = sum(i in test_data[test_data[:, 0] == u][:, 1] for i in top_k)
        precisions.append(n_rel_and_rec_k / 10)
    precision = sum(precisions) / len(precisions)

    print(f"Precision@10: {precision}")
