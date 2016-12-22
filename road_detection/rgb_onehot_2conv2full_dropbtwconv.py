import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
from datetime import timedelta, datetime
import math
import os
import scipy.misc as misc

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt

tf.__version__

if __name__ == "__main__":

    base_dir = r'F:\worldview2\Basic, Dallas, ,USA, 40cm_053951940010\ml_datasets\training\grayscale'
    base_dir = r'D:\data\machine_learning\worldview\dallas\grayscale'
    # base_dir = r'C:\dev\data\cnn_deconv'
    # base_dir = r'D:\data\machine_learning\worldview\dallas\rgb_onehot'
    base_dir = '/home/ec2-user/src/data/wv2/rgb_onehot'

    save_dir = os.path.join(base_dir, 'checkpoints_' + '2_conv_2_full_drop_btw_conv')
    save_path = os.path.join(save_dir, '20161212')
    # Create the directory if it does not exist.

    # In[38]:

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    in_dir = base_dir
    true_files = []
    true_targets = []
    false_files = []
    false_targets = []

    # Grab the true
    for dirpath, dirnames, filenames in os.walk(in_dir):
        for filename in [f for f in filenames if f.startswith("true")]:
            true_files.append(os.path.join(dirpath, filename))
            true_targets.append(1)

    # Grab the false data
    for dirpath, dirnames, filenames in os.walk(in_dir):
        for filename in [f for f in filenames
                         if (f.startswith("false"))]:
            false_files.append(os.path.join(dirpath, filename))
            false_targets.append(0)

    # Ensure they're both in the same order
    true_files.sort()
    false_files.sort()

    print(len(true_files))
    print(len(false_files))

    all_files = true_files + false_files
    all_targets = true_targets + false_targets
    shuffle_files, shuffle_targets = shuffle(all_files, all_targets)

    train_files, test_files, train_targets, test_targets = train_test_split(
        shuffle_files, shuffle_targets, train_size=0.90,
        stratify=shuffle_targets)

    img_size = 96

    # Load in the train images
    images_train = np.zeros((len(train_files), img_size, img_size, 3),
                            dtype='uint8')
    for ix, f in enumerate(train_files):
        images_train[ix, :, :, :] = misc.imread(f)

    y = np.array(train_targets).astype('uint8')
    labels_train = np.zeros((len(y), 2), dtype='uint8')
    for ix, y_val in enumerate(y):
        labels_train[ix, y[ix]] = 1

    cls_train = labels_train.argmax(1)

    # Load in the test images
    images_test = np.zeros((len(test_files), img_size, img_size, 3),
                           dtype='uint8')
    for ix, f in enumerate(test_files):
        images_test[ix, :, :, :] = misc.imread(f)

    y = np.array(test_targets).astype('uint8')
    labels_test = np.zeros((len(y), 3), dtype='uint8')
    for ix, y_val in enumerate(y):
        labels_test[ix, y[ix]] = 1

    cls_test = labels_test.argmax(1)

    num_channels = images_train.shape[-1]

    num_classes = 2

    # The CIFAR-10 data-set has now been loaded and consists of 60,000 images and associated labels (i.e. classifications of the images). The data-set is split into 2 mutually exclusive sub-sets, the training-set and the test-set.

    # In[10]:

    print("Size of:")
    print("- Training-set:\t\t{}".format(len(images_train)))
    print("- Test-set:\t\t{}".format(len(images_test)))

    img_size_cropped = img_size

    # ### Placeholder variables

    # Placeholder variables serve as the input to the TensorFlow computational graph that we may change each time we execute the graph. We call this feeding the placeholder variables and it is demonstrated further below.
    #
    # First we define the placeholder variable for the input images. This allows us to change the images that are input to the TensorFlow graph. This is a so-called tensor, which just means that it is a multi-dimensional array. The data-type is set to `float32` and the shape is set to `[None, img_size, img_size, num_channels]`, where `None` means that the tensor may hold an arbitrary number of images with each image being `img_size` pixels high and `img_size` pixels wide and with `num_channels` colour channels.

    # In[16]:

    x = tf.placeholder(tf.float32,
                       shape=[None, img_size, img_size, num_channels], name='x')

    # Next we have the placeholder variable for the true labels associated with the images that were input in the placeholder variable `x`. The shape of this placeholder variable is `[None, num_classes]` which means it may hold an arbitrary number of labels and each label is a vector of length `num_classes` which is 10 in this case.

    # In[17]:

    y_true = tf.placeholder(tf.float32, shape=[None, num_classes],
                            name='y_true')

    # We could also have a placeholder variable for the class-number, but we will instead calculate it using argmax. Note that this is a TensorFlow operator so nothing is calculated at this point.

    # In[18]:

    y_true_cls = tf.argmax(y_true, dimension=1)


    # ### Helper-function for creating Pre-Processing

    # The following helper-functions create the part of the TensorFlow computational graph that pre-processes the input images. Nothing is actually calculated at this point, the function merely adds nodes to the computational graph for TensorFlow.
    #
    # The pre-processing is different for training and testing of the neural network:
    # * For training, the input images are randomly cropped, randomly flipped horizontally, and the hue, contrast and saturation is adjusted with random values. This artificially inflates the size of the training-set by creating random variations of the original input images. Examples of distorted images are shown further below.
    #
    # * For testing, the input images are cropped around the centre and nothing else is adjusted.

    # In[19]:

    def pre_process_image(image, training):
        # This function takes a single image as input,
        # and a boolean whether to build the training or testing graph.

        if training:
            # For training, add the following to the TensorFlow graph.

            # Randomly crop the input image.
            image = tf.random_crop(image,
                                   size=[img_size_cropped, img_size_cropped,
                                         num_channels])

            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

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


    # The function above is called for each image in the input batch using the following function.

    # In[20]:

    def pre_process(images, training):
        # Use TensorFlow to loop over all the input images and call
        # the function above which takes a single image as input.
        images = tf.map_fn(lambda image: pre_process_image(image, training),
                           images)

        return images


    # ### Helper-function for creating Main Processing

    # The following helper-function creates the main part of the convolutional neural network. It uses Pretty Tensor which was described in the previous tutorials.

    # In[22]:

    def main_network(images, training):
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
                dropout(1.0). \
                conv2d(kernel=3, depth=3, name='layer_conv1',
                       batch_normalize=True). \
                max_pool(kernel=2, stride=2). \
                dropout(0.9). \
                conv2d(kernel=3, depth=3, name='layer_conv2',
                       batch_normalize=True). \
                max_pool(kernel=2, stride=2). \
                flatten(). \
                fully_connected(size=1024, name='layer_fc1'). \
                fully_connected(size=256, name='layer_fc2'). \
                softmax_classifier(2, labels=y_true)

        return y_pred, loss


    # ### Helper-function for creating Neural Network

    # The following helper-function creates the full neural network, which consists of the pre-processing and main-processing defined above.
    #
    # Note that the neural network is enclosed in the variable-scope named 'network'. This is because we are actually creating two neural networks in the TensorFlow graph. By assigning a variable-scope like this, we can re-use the variables for the two neural networks, so the variables that are optimized for the training-network are re-used for the other network that is used for testing.

    # In[23]:

    def create_network(training):
        # Wrap the neural network in the scope named 'network'.
        # Create new variables during training, and re-use during testing.
        with tf.variable_scope('network', reuse=not training):
            # Just rename the input placeholder variable for convenience.
            images = x

            # Create TensorFlow graph for pre-processing.
            images = pre_process(images=images, training=training)

            # Create TensorFlow graph for the main processing.
            y_pred, loss = main_network(images=images, training=training)

        return y_pred, loss


    # ### Create Neural Network for Training Phase

    # First create a TensorFlow variable that keeps track of the number of optimization iterations performed so far. In the previous tutorials this was a Python variable, but in this tutorial we want to save this variable with all the other TensorFlow variables in the checkpoints.
    #
    # Note that `trainable=False` which means that TensorFlow will not try to optimize this variable.

    # In[24]:

    global_step = tf.Variable(initial_value=0,
                              name='global_step', trainable=False)

    # Create the neural network to be used for training. The `create_network()` function returns both `y_pred` and `loss`, but we only need the `loss`-function during training.

    # In[25]:

    _, loss = create_network(training=True)

    # Create an optimizer which will minimize the `loss`-function. Also pass the `global_step` variable to the optimizer so it will be increased by one after each iteration.

    # In[26]:

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss,
                                                                    global_step=global_step)

    # ### Create Neural Network for Test Phase / Inference

    # Now create the neural network for the test-phase. Once again the `create_network()` function returns the predicted class-labels `y_pred` for the input images, as well as the `loss`-function to be used during optimization. During testing we only need `y_pred`.

    # In[27]:

    y_pred, _ = create_network(training=False)

    # We then calculate the predicted class number as an integer. The output of the network `y_pred` is an array with 10 elements. The class number is the index of the largest element in the array.

    # In[28]:

    y_pred_cls = tf.argmax(y_pred, dimension=1)

    # Then we create a vector of booleans telling us whether the predicted class equals the true class of each image.

    # In[29]:

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)

    # The classification accuracy is calculated by first type-casting the vector of booleans to floats, so that False becomes 0 and True becomes 1, and then taking the average of these numbers.

    # In[30]:

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # ### Saver
    #
    # In order to save the variables of the neural network, so they can be reloaded quickly without having to train the network again, we now create a so-called Saver-object which is used for storing and retrieving all the variables of the TensorFlow graph. Nothing is actually saved at this point, which will be done further below.

    # In[31]:

    saver = tf.train.Saver()

    # ## TensorFlow Run

    # ### Create TensorFlow session
    #
    # Once the TensorFlow graph has been created, we have to create a TensorFlow session which is used to execute the graph.

    # In[36]:

    session = tf.Session()

    # First try to restore the latest checkpoint. This may fail and raise an exception e.g. if such a checkpoint does not exist, or if you have changed the TensorFlow graph.

    # In[40]:

    try:
        print("Trying to restore last checkpoint ...")

        # Use TensorFlow to find the latest checkpoint - if any.
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

        # Try and load the data in the checkpoint.
        saver.restore(session, save_path=last_chk_path)

        # If we get to this point, the checkpoint was successfully loaded.
        print("Restored checkpoint from:", last_chk_path)
    except:
        # If the above failed for some reason, simply
        # initialize all the variables for the TensorFlow graph.
        print("Failed to restore checkpoint. Initializing variables instead.")
        session.run(tf.global_variables_initializer())

    # ### Helper-function to get a random training-batch

    # There are 50,000 images in the training-set. It takes a long time to calculate the gradient of the model using all these images. We therefore only use a small batch of images in each iteration of the optimizer.
    #
    # If your computer crashes or becomes very slow because you run out of RAM, then you may try and lower this number, but you may then need to perform more optimization iterations.

    # In[41]:

    train_batch_size = 256


    # Function for selecting a random batch of images from the training-set.

    # In[42]:

    def random_batch():
        # Number of images in the training-set.
        num_images = len(images_train)

        # Create a random index.
        idx = np.random.choice(num_images,
                               size=train_batch_size,
                               replace=False)

        # Use the random index to select random images and labels.
        x_batch = images_train[idx, :, :, :]
        y_batch = labels_train[idx, :]

        return x_batch, y_batch


    # ### Helper-function to perform optimization

    # This function performs a number of optimization iterations so as to gradually improve the variables of the network layers. In each iteration, a new batch of data is selected from the training-set and then TensorFlow executes the optimizer using those training samples.  The progress is printed every 100 iterations. A checkpoint is saved every 1000 iterations and also after the last iteration.

    # In[43]:

    def optimize(num_iterations):
        # Start-time used for printing time-usage below.
        start_time = time.time()

        for i in range(num_iterations):
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = random_batch()

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            # We also want to retrieve the global_step counter.
            i_global, _ = session.run([global_step, optimizer],
                                      feed_dict=feed_dict_train)

            # Print status to screen every 100 iterations (and last).
            if (i_global % 100 == 0) or (i == num_iterations - 1):
                # Calculate the accuracy on the training-batch.
                batch_acc = session.run(accuracy,
                                        feed_dict=feed_dict_train)

                # Print status.
                msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
                print(msg.format(i_global, batch_acc))

            # Save a checkpoint to disk every 1000 iterations (and last).
            if (i_global % 1000 == 0) or (i == num_iterations - 1):
                # Save all variables of the TensorFlow graph to a
                # checkpoint. Append the global_step counter
                # to the filename so we save the last several checkpoints.
                saver.save(session,
                           save_path=save_path,
                           global_step=global_step)

                print("Saved checkpoint.")

        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


    # ### Helper-functions for calculating classifications
    #
    # This function calculates the predicted classes of images and also returns a boolean array whether the classification of each image is correct.
    #
    # The calculation is done in batches because it might use too much RAM otherwise. If your computer crashes then you can try and lower the batch-size.

    # In[46]:

    # Split the data-set in batches of this size to limit RAM usage.
    batch_size = 256


    def predict_cls(images, labels, cls_true):
        # Number of images.
        num_images = len(images)

        # Allocate an array for the predicted classes which
        # will be calculated in batches and filled into this array.
        cls_pred = np.zeros(shape=num_images, dtype=np.int)

        # Now calculate the predicted classes for the batches.
        # We will just iterate through all the batches.
        # There might be a more clever and Pythonic way of doing this.

        # The starting index for the next batch is denoted i.
        i = 0

        while i < num_images:
            # The ending index for the next batch is denoted j.
            j = min(i + batch_size, num_images)

            # Create a feed-dict with the images and labels
            # between index i and j.
            feed_dict = {x: images[i:j, :],
                         y_true: labels[i:j, :]}

            # Calculate the predicted class using TensorFlow.
            cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

            # Set the start-index for the next batch to the
            # end-index of the current batch.
            i = j

        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)

        return correct, cls_pred


    # Calculate the predicted class for the test-set.

    # In[47]:

    def predict_cls_test():
        return predict_cls(images=images_test,
                           labels=labels_test,
                           cls_true=cls_test)


    # ### Helper-functions for the classification accuracy
    #
    # This function calculates the classification accuracy given a boolean array whether each image was correctly classified. E.g. `classification_accuracy([True, True, False, False, False]) = 2/5 = 0.4`. The function also returns the number of correct classifications.

    # In[48]:

    def classification_accuracy(correct):
        # When averaging a boolean array, False means 0 and True means 1.
        # So we are calculating: number of True / len(correct) which is
        # the same as the classification accuracy.

        # Return the classification accuracy
        # and the number of correct classifications.
        return correct.mean(), correct.sum()


    # ### Helper-function for showing the performance

    # Function for printing the classification accuracy on the test-set.
    #
    # It takes a while to compute the classification for all the images in the test-set, that's why the results are re-used by calling the above functions directly from this function, so the classifications don't have to be recalculated by each function.

    # In[49]:

    def print_test_accuracy(show_example_errors=False,
                            show_confusion_matrix=False):

        # For all the images in the test-set,
        # calculate the predicted classes and whether they are correct.
        correct, cls_pred = predict_cls_test()

        # Classification accuracy and the number of correct classifications.
        acc, num_correct = classification_accuracy(correct)

        # Number of images being classified.
        num_images = len(correct)

        # Print the accuracy.
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, num_correct, num_images))


    # ## Perform optimization

    # My laptop computer is a Quad-Core with 2 GHz per core. It has a GPU but it is not fast enough for TensorFlow so it only uses the CPU. It takes about 1 hour to perform 10,000 optimization iterations using the CPU on this PC. For this tutorial I performed 150,000 optimization iterations so that took about 15 hours. I let it run during the night and at various points during the day.
    #
    # Because we are saving the checkpoints during optimization, and because we are restoring the latest checkpoint when restarting the code, we can stop and continue the optimization later.

    # In[56]:

    optimize(10000)
