import os
import time

from datetime import datetime, timedelta

import numpy as np
import scipy.misc as misc
import tensorflow as tf

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt

print('TensorFlow version: ' + tf.__version__)


class RoadDetectCNN(object):
    raw_files = []
    truth_files = []
    num_classes = None
    image_size = None
    img_size_flat = None
    x = None
    y_true = None
    saver = None
    global_step = None
    optimizer = None
    session = None

    def __init__(self, base_dir, save_path=None):
        self.base_dir = base_dir
        if save_path is None:
            self.save_dir = os.path.join(base_dir, 'checkpoints')
            save_path = os.path.join(self.save_dir,
                                     datetime.now().strftime('%Y%m%d_%H%M%S'))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save_path = save_path

    def load_data(self):
        in_dir = self.base_dir

        # Grab the raw data files with the point spread functions
        for dirpath, dirnames, filenames in os.walk(in_dir):
            for filename in [f for f in filenames if f.startswith("raw")]:
                self.raw_files.append(os.path.join(dirpath, filename))

        # Grab the raw data files with the truth locations
        for dirpath, dirnames, filenames in os.walk(in_dir):
            for filename in [f for f in filenames
                             if (f.startswith("truth"))]:
                self.truth_files.append(os.path.join(dirpath, filename))

        # Ensure they're both in the same order
        self.raw_files.sort()
        self.truth_files.sort()

    def set_train_test(self, train_pct=0.9):
        self.image_size = 128
        ntrain = int(train_pct * len(self.raw_files))
        # ntrain = 5000
        train_inds = np.random.choice(np.arange(len(self.raw_files)), ntrain,
                                      replace=False)
        test_inds = np.setdiff1d(np.arange(len(self.raw_files)), train_inds)
        # test_inds = np.random.choice(test_inds, 200, replace=False)
        ntest = len(test_inds)

        train_files = np.array(self.raw_files)[train_inds].tolist()
        train_targets = np.array(self.truth_files)[train_inds].tolist()
        test_files = np.array(self.raw_files)[test_inds].tolist()
        test_targets = np.array(self.truth_files)[test_inds].tolist()

        # Load in the train images
        images_train = np.zeros((len(train_files), self.image_size,
                                 self.image_size, 1),
                                dtype='uint8')
        for ix, f in enumerate(train_files):
            images_train[ix, :, :, 0] = misc.imread(f)

        # Load in the train truth
        truth_train = np.zeros((len(train_targets),
                                self.image_size * self.image_size),
                               dtype='uint8')
        for ix, f in enumerate(train_targets):
            truth_train[ix, :] = misc.imread(f).flatten()

        # Load in the test images
        images_test = np.zeros((len(test_files), self.image_size,
                                self.image_size, 1),
                               dtype='uint8')
        for ix, f in enumerate(test_files):
            images_test[ix, :, :, 0] = misc.imread(f)

        # Load in the test truth
        truth_test = np.zeros((len(test_targets),
                               self.image_size * self.image_size),
                              dtype='uint8')
        for ix, f in enumerate(test_targets):
            truth_test[ix, :] = misc.imread(f).flatten()

        cls_train = truth_train
        labels_train = truth_train

        cls_test = truth_test
        labels_test = truth_test

        num_channels = images_train.shape[-1]

        self.num_classes = self.image_size * self.image_size

        # Images are stored in one-dimensional arrays of this length.
        self.img_size_flat = self.image_size * self.image_size

        print("Size of:")
        print("- Training-set:\t\t{}".format(len(images_train)))
        print("- Test-set:\t\t{}".format(len(images_test)))

        return images_train, images_test, truth_train, truth_test

    def main_network(self, images, training=False):
        # Wrap the input images as a Pretty Tensor object.
        x_pretty = pt.wrap(images)
        # Pretty Tensor uses special numbers to distinguish between
        # the training and testing phases.
        if training:
            phase = pt.Phase.train
        else:
            phase = pt.Phase.infer

        # Create the convolutional neural network using Pretty Tensor.
        # It is very similar to the previous tutorials, except
        # the use of so-called batch-normalization in the first layer.
        with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
            y_pred, loss = x_pretty. \
                dropout(0.8). \
                conv2d(kernel=3, depth=8,
                       name='layer_conv1',
                       batch_normalize=True). \
                max_pool(kernel=2, stride=2). \
                dropout(0.8). \
                conv2d(kernel=3, depth=16,
                       name='layer_conv2',
                       batch_normalize=True). \
                max_pool(kernel=2, stride=2). \
                dropout(0.8). \
                conv2d(kernel=5, depth=32,
                       name='layer_conv3',
                       batch_normalize=True). \
                max_pool(kernel=2, stride=2). \
                dropout(0.8). \
                flatten(). \
                fully_connected(size=10000, name='layer_fc1'). \
                softmax_classifier(self.num_classes, labels=self.y_true)

        return y_pred, loss

    def create_main_network(self, images, num_classes, training=False):
        # Wrap the neural network in the scope named 'network'.
        # Create new variables during training, and re-use during testing.

        img_size = images.shape

        self.x = tf.placeholder(tf.float32,
                                shape=[None, img_size[0], img_size[1], 1],
                                name='x')

        # Labels for the images in 'x'. The None in shape allows for
        # arbitrary size of labels
        self.y_true = tf.placeholder(tf.float32, shape=[None, num_classes],
                                     name='y_true')

        with tf.variable_scope('network', reuse=not training):
            # Just rename the input placeholder variable for convenience.
            images = self.x

            # Create TensorFlow graph for pre-processing.
            # images = self.pre_process(images=images, training=training)

            # Create TensorFlow graph for the main processing.
            y_pred, loss = self.main_network(images=images, training=training)

        return y_pred, loss

    def fcn_network(self, images, training=True):
        """
        This network implementation uses a convolutional layer as the final
        layer prior to the softmax to allow for the input of an image of any
        dimensions. Otherwise when you use a traditionally fully connected
        layer you will be restricted to a specific size.
        :param images: The images to use for training or testing
        :param training: Whether or not we want to set up the training phase
        of the network or the testing phase of the network
        :return:
        """
        # Wrap the input images as a Pretty Tensor object
        x_pretty = pt.wrap(images)
        # Pretty Tensor uses special numbers to distinguish between
        # the training and testing phases.
        if training:
            phase = pt.Phase.train
        else:
            phase = pt.Phase.infer

        """
        An example of the number of neurons as each layer progresses
        Image Size: 128 x 128
        Conv2d: 3x3x8
        """
        with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
            y_pred, loss = x_pretty. \
                conv2d(kernel=3, depth=8,
                       name='layer_conv1',
                       batch_normalize=True). \
                max_pool(kernel=2, stride=2). \
                conv2d(kernel=3, depth=16,
                       name='layer_conv2',
                       batch_normalize=True). \
                max_pool(kernel=2, stride=2). \
                conv2d(kernel=5, depth=32,
                       name='layer_conv3',
                       batch_normalize=True). \
                max_pool(kernel=2, stride=2). \
                conv2d(kernel=128, depth=1,
                       name='fully_connected_layer',
                       batch_normalize=True). \
                softmax_classifier(self.num_classes, labels=self.y_true)

        return y_pred, loss

    def create_fcn_network(self, images, num_classes, training=False):
        # Wrap the neural network in the scope named 'network'.
        # Create new variables during training, and re-use during testing.

        img_size = images.shape

        self.x = tf.placeholder(tf.float32,
                                shape=[None, img_size[0], img_size[1], 1],
                                name='x')

        # Labels for the images in 'x'. The None in shape allows for
        # arbitrary size of labels
        self.y_true = tf.placeholder(tf.float32, shape=[None, num_classes],
                                     name='y_true')

        with tf.variable_scope('network', reuse=not training):
            # Just rename the input placeholder variable for convenience.
            images = self.x

            # Create TensorFlow graph for pre-processing.
            # images = self.pre_process(images=images, training=training)

            # Create TensorFlow graph for the main processing.
            y_pred, loss = self.main_network(images=images, training=training)

        return y_pred, loss

    def pre_process_image(self, image, training=False):
        # This function takes a single image as input,
        # and a boolean whether to build the training or testing graph.

        if training:
            # For training, add the following to the TensorFlow graph.

            # Randomly crop the input image.
            image = tf.random_crop(image, size=[self.img_size, self.img_size,
                                                self.num_channels])

            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)
            image = tf.image.flip_up_down(image)

            # Randomly adjust hue, contrast and saturation.
            # image = tf.image.random_hue(image, max_delta=0.05)
            # image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            # image = tf.image.random_brightness(image, max_delta=0.2)
            # image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

            # Some of these functions may overflow and result in pixel
            # values beyond the [0, 1] range. It is unclear from the
            # documentation of TensorFlow 0.10.0rc0 whether this is
            # intended. A simple solution is to limit the range.

            # Limit the image pixels between [0, 1] in case of overflow.
            # image = tf.minimum(image, 1.0)
            # image = tf.maximum(image, 0.0)
        else:
            # For training, add the following to the TensorFlow graph.

            # Crop the input image around the centre so it is the same
            # size as images that are randomly cropped during training.
            image = tf.image.resize_image_with_crop_or_pad(image,
                                                           target_height=img_size_cropped,
                                                           target_width=img_size_cropped)

        return image

    def pre_process(self, images, training=False):
        # Use TensorFlow to loop over all the input images and call
        # the function above which takes a single image as input.
        images = tf.map_fn(lambda image: self.pre_process_image(image,
                                                                training),
                           images)

        return images

    def train_main(self, images, labels, batch_size=64, gpu_count=0):
        # First create a TensorFlow variable that keeps track of the number of
        # optimization iterations performed so far. In the previous tutorials
        # this was a Python variable, but in this tutorial we want to save this
        # variable with all the other TensorFlow variables in the checkpoints.
        # Note that `trainable=False` which means that TensorFlow will not try
        # to optimize this variable.
        self.global_step = tf.Variable(initial_value=0,
                                       name='global_step', trainable=False)

        # Create the neural network to be used for training.
        # The `create_network()` function returns both `y_pred` and `loss`, but
        # we only need the `loss`-function during training.
        _, loss = self.create_main_network(images,
                                           np.product(labels.shape),
                                           training=True)

        # Create an optimizer which will minimize the `loss`-function.
        # Also pass the `global_step` variable to the optimizer so it will be
        # increased by one after each iteration.
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.optimizer = self.optimizer.minimize(loss,
                                                 global_step=self.global_step)

        # ### Create Neural Network for Test Phase / Inference
        # Now create the neural network for the test-phase.
        # Once again the `create_network()` function returns the predicted
        # class-labels `y_pred` for the input images, as well as the
        # `loss`-function to be used during optimization. During testing we only
        # need `y_pred`.
        y_pred, _ = self.create_main_network(training=False)

        # Then we create a vector of booleans telling us whether the predicted
        # class equals the true class of each image.
        correct_prediction = tf.equal(y_pred, labels)

        # The classification accuracy is calculated by first type-casting the
        # vector of booleans to floats, so that False becomes 0 and True
        # becomes 1, and then taking the average of these numbers.
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # In order to save the variables of the neural network, so they can be
        # reloaded quickly without having to train the network again, we now
        # create a so-called Saver-object which is used for storing and
        # retrieving all the variables of the TensorFlow graph.
        # Nothing is actually saved at this point, which will be done further
        # below.
        self.saver = tf.train.Saver()

        # Once the TensorFlow graph has been created, we have to create a
        # TensorFlow session which is used to execute the graph.
        # Change GPU value to 1 if you want to use the GPU, I was having
        # problems with it working for me.
        session_config = tf.ConfigProto(device_count={'GPU': gpu_count})

        self.session = tf.Session(config=session_config)

        # First try to restore the latest checkpoint.
        print("Trying to restore last checkpoint ...")
        # Use TensorFlow to find the latest checkpoint - if any.
        last_chk_path = tf.train.latest_checkpoint(
            checkpoint_dir=self.save_dir)

        if last_chk_path is not None:
            # Try and load the data in the checkpoint.
            self.saver.restore(self.session, save_path=last_chk_path)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)
        else:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint. "
                  "Initializing variables instead.")
            self.session.run(tf.global_variables_initializer())

        self.optimize(images, labels, batch_size, num_iterations=1000)

    def optimize(self, images, labels, batch_size, num_iterations):
        """
        This function performs a number of optimization iterations so as to
        gradually improve the variables of the network layers.
        In each iteration, a new batch of data is selected from the
        training-set and then TensorFlow executes the optimizer using those
        training samples.  The progress is printed every 100 iterations.
        A checkpoint is saved every 1000 iterations and also after the last
        iteration.
        :param images: The images to train on
        :param labels: The labels for the images (should match the size of
        images)
        :param batch_size: The size of the batch to reduce our memory
        footprint. A good value is 64 - 256 just to be safe. It will train
        fast the larger this number.
        :param num_iterations: The number of iterations before it stops learning
        :return:
        """
        # Start-time used for printing time-usage below.
        start_time = time.time()

        for i in range(num_iterations):
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = random_batch(images, labels, batch_size)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {self.x: x_batch,
                               self.y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            # We also want to retrieve the global_step counter.
            i_global, _ = self.session.run([self.global_step, self.optimizer],
                                           feed_dict=feed_dict_train)

            # Print status to screen every 100 iterations (and last).
            if (i_global % 100 == 0) or (i == num_iterations - 1):
                # Calculate the accuracy on the training-batch.
                batch_acc = self.session.run(self.accuracy,
                                             feed_dict=feed_dict_train)

                # Print status.
                msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
                print(msg.format(i_global, batch_acc))

            # Save a checkpoint to disk every 1000 iterations (and last).
            if (i_global % 1000 == 0) or (i == num_iterations - 1):
                # Save all variables of the TensorFlow graph to a
                # checkpoint. Append the global_step counter
                # to the filename so we save the last several checkpoints.
                self.saver.save(self.session, save_path=self.save_dir,
                                global_step=self.global_step)

                print("Saved checkpoint.")

        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_diff = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))


# Function for selecting a random batch of images from the training-set.
def random_batch(images, labels, batch_size):
    # Number of images in the training-set.
    num_images = len(images)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images[idx, :, :, :]
    y_batch = labels[idx, :]

    return x_batch, y_batch
