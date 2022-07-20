def read_labeled_tfrecord_example(example):
    features = {
        'rbg': tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring
        'hsv': tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        'label': tf.io.FixedLenFeature([], tf.int64),  # one bytestring
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)
    # FixedLenFeature fields are now ready to use: exmple['size']
    # VarLenFeature fields require additional sparse_to_dense decoding
    
    lbp83 = example['hsv']
    image = tf.image.decode_jpeg(example['rbg'], channels=3)
    image = tf.reshape(image, [IMAGE_SIZE[0],IMAGE_SIZE[0],3])
    label = tf.cast(example['label'], tf.int32)
    return image, lbp83, label

def load_dataset_example(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord_example, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def get_training_dataset_example():
    dataset = load_dataset_example(TRAINING_FILENAMES, labeled=True)
    # dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.shuffle(2048)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset