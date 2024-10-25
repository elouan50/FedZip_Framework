import collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff

from config import *
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow import reshape, nest, config


def build_train_tff_data():
    x_train, y_train = tfds.as_numpy(tfds.load(name_dt,
                                               split='train',
                                               batch_size=-1,
                                               as_supervised=True,
                                               ))
    shape = x_train[1].shape
    
    if input_shape != shape:
        x_train = np.array([resize_with_pad(img, input_shape[1], input_shape[2]) for img in x_train])

    x_train = x_train.reshape(input_shape)
    y_train = to_categorical(y_train, NumClass)

    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.int32)

    total_image_count = len(x_train)
    image_per_set = int(np.floor(total_image_count / Num_Client))

    client_train_dataset = collections.OrderedDict()
    for i in range(1, Num_Client + 1):
        client_name = "client_" + str(i)
        start = image_per_set * (i - 1)
        end = image_per_set * i

        # print(f"Adding data from {start} to {end} for client : {client_name}")
        data = collections.OrderedDict((('label', y_train[start:end]), ('pixels', x_train[start:end])))
        client_train_dataset[client_name] = data

    train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)
    # print(train_dataset)

    sample_dataset = train_dataset.create_tf_dataset_for_client(train_dataset.client_ids[0])
    sample_element = next(iter(sample_dataset))

    SHUFFLE_BUFFER = image_per_set
    PREFETCH_BUFFER = 10

    def preprocess(dataset):
        def batch_format_fn(element):
            """Flatten a batch `pixels` and return the features as an `OrderedDict`."""

            return collections.OrderedDict(
                x=reshape(element['pixels'], input_shape),
                y=reshape(element['label'], [-1, NumClass]))

        return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
            BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)

    #
    preprocessed_sample_dataset = preprocess(sample_dataset)
    sample_batch = nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_sample_dataset)))

    def make_federated_data(client_data, client_ids):
        return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

    federated_train_data = make_federated_data(train_dataset, train_dataset.client_ids)

    print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
    print('First dataset: {d}'.format(d=federated_train_data[0]))

    return federated_train_data, preprocessed_sample_dataset

def resize_with_pad(image, target_height, target_width):
    h, w = image.shape[:2]
    pad_h = max(target_height - h, 0)
    pad_w = max(target_width - w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)


def build_test_data():
    x_test, y_test = tfds.as_numpy(tfds.load(name_dt,
                                             split='test',
                                             batch_size=-1,
                                             as_supervised=True,
                                             ))

    shape = x_test[1].shape

    if input_shape != shape:
        x_test = np.array([resize_with_pad(img, input_shape[1], input_shape[2]) for img in x_test])

    x_test = x_test.reshape(input_shape)
    y_test = to_categorical(y_test, NumClass)

    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.int32)

    return x_test, y_test


def load_dataset(phase='train'):
    """
    :param phase: it should be 'test' or 'train'

    :return: loding_dataset the dataset in federated mode for Training, or
             simple sequence dataset for Testing Phase
    """

    if phase == 'train':
        return build_train_tff_data()
    elif phase == 'test':
        return build_test_data()
