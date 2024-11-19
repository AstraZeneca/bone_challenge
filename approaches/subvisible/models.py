import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
import numpy as np
import pandas as pd
import nibabel as nib
import glob
import pathlib
from skimage.transform import resize
import time
import warnings
warnings.filterwarnings('ignore')

IMG_HEIGHT, IMG_WIDTH = 96, 96
MIN_INT, MAX_INT = -1417, 3171
BATCH_SIZE = 32
EPOCHS = 120

class ImageClassifier:
    def __init__(self, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, batch_size=BATCH_SIZE, epochs=EPOCHS):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.epochs = epochs
        self.AUTOTUNE = tf.data.AUTOTUNE
        
    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.class_names
        return tf.argmax(one_hot)

    def process_path_trn_cls(self, file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = tf.io.decode_image(img)
        img = tf.image.resize_with_crop_or_pad(img, self.img_height, self.img_width)
        img = tf.image.rot90(img, tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32))
        img = tf.image.random_flip_left_right(img)
        return img, label

    def process_path_tst_cls(self, file_path):
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = tf.io.decode_image(img)
        img = tf.image.resize_with_crop_or_pad(img, self.img_height, self.img_width)
        return img, label

    def configure_for_performance(self, ds):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds
    
    def create_cls_model(self):
        model = Sequential([
            layers.Input(shape=(self.img_height, self.img_width, 1)),
            layers.Conv2D(32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.AveragePooling2D(),
            layers.Conv2D(32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.AveragePooling2D(),
            layers.Conv2D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            layers.GlobalAveragePooling2D(),
            layers.Flatten(),
            layers.Dense(2)
        ])
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        return model
    
    def train_cls(self, trn_folder, val_folder, logdir, checkpoint_filepath):
        dir_trn = pathlib.Path(trn_folder)
        dir_val = pathlib.Path(val_folder)
        trn_ds = tf.data.Dataset.list_files(str(dir_trn/'*/*.png'))
        val_ds = tf.data.Dataset.list_files(str(dir_val/'*/*.png'))
        self.class_names = np.array(sorted([item.name for item in dir_val.glob('*')]))
        trn_ds = trn_ds.map(self.process_path_trn_cls)
        val_ds = val_ds.map(self.process_path_tst_cls)
        trn_ds = self.configure_for_performance(trn_ds)
        val_ds = self.configure_for_performance(val_ds)
        self.model = self.create_cls_model()

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        file_writer = tf.summary.create_file_writer(logdir + "/metrics")
        file_writer.set_as_default()

        def lr_schedule(epoch):
            learning_rate = 0.001
            if epoch > 80:
                learning_rate = 0.0001

            tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
            return learning_rate

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            save_best_only=True
        )

        history = self.model.fit(
            trn_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=[model_checkpoint_callback, tensorboard_callback, lr_callback]
        )

def create_reg_model():
    model = Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 51)),
        layers.SeparableConv2D(32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)), 
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.AveragePooling2D(),
        layers.Conv2D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss=custom_loss)
    return model


def calculate_score(pred_slice_num, gt_slice_num):
    """Returns the survival function of a single-sided normal distribution with stddev=3."""
    pred_slice_num = tf.cast(pred_slice_num, tf.float32)
    gt_slice_num = tf.cast(gt_slice_num, tf.float32)
    diff = tf.abs(pred_slice_num - gt_slice_num)
    stddev = 3.0
    score = 2 * 0.5 * tf.math.erfc(diff / (stddev * np.sqrt(2.0)))
    return score


class CustomScoreMetric(tf.keras.metrics.Metric):
    def __init__(self, name='custom_score', **kwargs):
        super(CustomScoreMetric, self).__init__(name=name, **kwargs)
        self.custom_score = self.add_weight(name='cs', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        score = calculate_score(y_pred, y_true)
        self.custom_score.assign(tf.reduce_mean(score))

    def result(self):
        return self.custom_score

    def reset_states(self):
        self.custom_score.assign(0.0)


def custom_loss(y_true, y_pred):
    score = calculate_score(y_pred, y_true)
    return -tf.reduce_mean(score)


def process_trn(img, label):
    img = tf.image.rot90(img, tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32))
    img = tf.image.random_flip_left_right(img)
    return img, label


def lr_schedule(epoch):
    learning_rate = 0.001
    if epoch > 80:
        learning_rate = 0.0005
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


def read_process(df):
    out = []
    for i in range(len(df)):
        image = nib.load(df.iloc[i]["image_path"])
        st_indx = df.iloc[i]["predicted GPI"]-25
        en_indx = df.iloc[i]["predicted GPI"]+26
        img_array = np.zeros((IMG_HEIGHT, IMG_WIDTH, 51))
        for indx, j in enumerate(range(st_indx, en_indx)):
            img = image.get_fdata()[:, :, j]
            img = (img - MIN_INT)/(MAX_INT - MIN_INT)
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH),  anti_aliasing=False)
            img_array[:, :, indx] = np.uint8(255*img)
        out.append(img_array)
    images = np.array(out)
    y = []
    for i in range(len(df)):
        y.append(df.iloc[i]["Growth Plate Index"])
    y = np.array(y)
    yp = []
    for i in range(len(df)):
        yp.append(df.iloc[i]["predicted GPI"])
    yp = np.array(yp)
    labels = 25+(y - yp)
    return images, labels


def get_init_estimate(image, model_cls):
    img_array = np.zeros((image.shape[-1], image.shape[0], image.shape[1], 1))
    for j in range(image.shape[-1]):
        img = image.get_fdata()[:, :, j]
        img = (img - MIN_INT)/(MAX_INT - MIN_INT)
        img = np.uint8(255*img)
        img = np.expand_dims(img, axis=(0, 3))
        img_array[j, :, :, :] = img
    img_array = tf.image.resize_with_crop_or_pad(img_array, IMG_HEIGHT, IMG_WIDTH)
    predictions = tf.nn.softmax(model_cls.predict(img_array, verbose=0))
    y = predictions[:, 1]
    indices = np.where(y == 1)
    if len(np.ravel(indices)) == 0:
        thr = 0.1
        while (len(np.ravel(indices)) == 0):
            indices = np.where(y > (1-thr))
            thr += 0.1
    pgpi_init = np.ravel(indices)[-1]
    return pgpi_init


def make_init_estimate(data, checkpoint_filepath):
    classifier = ImageClassifier()
    model_cls = classifier.create_cls_model()
    model_cls.load_weights(checkpoint_filepath)
    data["predicted GPI"] = 0
    for i in range(len(data)):
        image = nib.load(data.iloc[i]["image_path"])
        pgpi_init = get_init_estimate(image, model_cls)
        data.at[data.index[i], "predicted GPI"] = pgpi_init
        return data


def train(data, checkpoint_filepath, log_path):
    data = make_init_estimate(data, checkpoint_filepath)
    for fold in range(5):
        print('Training ------------------------------------- Fold ', fold)
        trn = data[data['fold'] != fold]
        tst = data[data['fold'] == fold]
        trn_img, trn_label = read_process(trn)
        tst_img, tst_label = read_process(tst)
        trn_ds = tf.data.Dataset.from_tensor_slices((trn_img, trn_label))
        val_ds = tf.data.Dataset.from_tensor_slices((tst_img, tst_label))
        trn_ds = (trn_ds.shuffle(50)
                        .map(process_trn)
                        .batch(BATCH_SIZE)
                        .prefetch(64))
        val_ds = (val_ds.batch(20).prefetch(20))
        model = create_reg_model()
        # Log directory for TensorBoard
        logdir = os.path.join(log_path, 'logs', f'Fold_{fold}')
        # Checkpoint directory
        checkpoint_filepath = os.path.join(log_path, 'checkpoints', f'Fold_{fold}')

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                        filepath=checkpoint_filepath,
                                        monitor='val_custom_score',
                                        save_weights_only=True,
                                        save_best_only=True,
                                        mode='max')
        file_writer = tf.summary.create_file_writer(logdir + "/metrics")
        file_writer.set_as_default()
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        history = model.fit(
                    trn_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks=[model_checkpoint_callback, lr_callback],
                    verbose=0)


def make_prediction(path, cls_checkpoint_filepath, reg_checkpoints_dir):
    fn = glob.glob(os.path.join(path, '*.nii'))
    pgpi = []
    names = []
    classifier = ImageClassifier()
    model_cls = classifier.create_cls_model()
    model_cls.load_weights(cls_checkpoint_filepath)
    model_reg = create_reg_model()
    for n in fn:
        print('-----------------')
        print(n)
        names.append(n.split('/')[-1])
        start_time = time.time()
        image = nib.load(n)
        pgpi_init = get_init_estimate(image, model_cls)
        st_indx = pgpi_init-25
        en_indx = pgpi_init+26
        img_array = np.zeros((IMG_HEIGHT, IMG_WIDTH, 51))

        for indx, j in enumerate(range(st_indx, en_indx)):
            img = image.get_fdata()[:, :, j]
            img = (img - MIN_INT)/(MAX_INT - MIN_INT)
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH),  anti_aliasing=False)
            img_array[:, :, indx] = np.uint8(255*img)

        img_array = np.expand_dims(img_array, axis=0)
        p = []
        for fold in range(5):
            checkpoint_filepath = os.path.join(reg_checkpoints_dir, f'Fold{fold}/')
            model_reg.load_weights(checkpoint_filepath)
            pred = model_reg.predict(img_array, verbose=0)
            p.append(pred[0][0])
        delta = np.round(np.mean(p))-25
        pgpi.append(pgpi_init+delta)
        end_time = time.time()
        print('Predicted Growth Plate Index: ', pgpi[-1], ' Prediction time in seconds: ', end_time - start_time)
    return pd.DataFrame({'image_name': names, 'predicted_index': pgpi})



