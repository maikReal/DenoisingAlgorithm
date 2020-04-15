import numpy as np
from pydub import AudioSegment
from scipy.io.wavfile import read
import os
import glob
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

from model import unet
import config as cnf

SAVE_WEIGHTS = cnf.save_weights
TRAIN_FOLDER = cnf.train_folder

CLEAN_TRAIN_DATA = os.path.join(TRAIN_FOLDER, "clean")
NOISY_TRAIN_DATA = os.path.join(TRAIN_FOLDER, "noisy")

TRAIN_PATHS = glob.glob(f"{CLEAN_TRAIN_DATA}/**/*.npy")
random.shuffle(TRAIN_PATHS)

weights_path = cnf.weights_path

batch_size = cnf.batch_size


def split_data(train_proportion=cnf.train_proportion):
    thr = int((train_proportion * len(TRAIN_PATHS)) / 100)

    train_clean_paths = TRAIN_PATHS[:thr]
    train_noisy_paths = [i.replace("clean", "noisy") for i in TRAIN_PATHS[:thr]]

    test_clean_paths = TRAIN_PATHS[thr:]
    test_noisy_paths = [i.replace("clean", "noisy") for i in TRAIN_PATHS[thr:]]

    return train_clean_paths, train_noisy_paths, test_clean_paths, test_noisy_paths


def data_generator(
    CLEAN_PATHS, NOISY_PATHS, sound_threshold=cnf.sound_threshold, batch_size=32
):

    for i in range(len(CLEAN_PATHS)):

        rand_indx = random.sample(range(len(CLEAN_PATHS)), batch_size)

        batch_x = []
        batch_y = []
        for i in rand_indx:
            x = np.load(NOISY_PATHS[i])
            y = np.load(CLEAN_PATHS[i])

            if x.shape[0] > sound_threshold:
                x = x[:sound_threshold, :]
                y = y[:sound_threshold, :]

            if x.shape[0] < sound_threshold:
                shape_diff = sound_threshold - x.shape[0]
                zero_mtrx = np.zeros((shape_diff, 80))

                x = np.concatenate((x, zero_mtrx))
                y = np.concatenate((y, zero_mtrx))

            batch_x.append(x)
            batch_y.append(y)

        x = np.array(batch_x)
        y = np.array(batch_y)

        y = x - y

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        y = y.reshape(y.shape[0], y.shape[1], x.shape[2], 1)

        yield x, y


def train_model():

    (
        train_clean_paths,
        train_noisy_paths,
        test_clean_paths,
        test_noisy_paths,
    ) = split_data()

    print("Train data:", len(train_clean_paths))
    print("Test data:", len(test_clean_paths))
    train_generator = data_generator(train_clean_paths, train_noisy_paths)
    test_generator = data_generator(test_clean_paths, test_noisy_paths)

    print("Initialize the model...")
    model = unet(pretrained_weights=weights_path)

    print("Start training...")
    model.fit_generator(
        generator=train_generator,
        validation_data=test_generator,
        steps_per_epoch=cnf.steps_per_epoch,
        epochs=cnf.epochs,
        validation_steps=cnf.validation_steps,
    )
    print("Finishing training...")

    if SAVE_WEIGHTS:
        model.save_weights(f"weights/{str(datetime.now())}_unet_model.h5")

    return model


if __name__ == "__main__":
    model = train_model()

    print("HERE")
    if cnf.CHECK_ON_VAL_DATA:
        VALIDATE_FOLDER = cnf.val_folder

        CLEAN_VAL = os.path.join(VALIDATE_FOLDER, "clean")
        NOISY_VAL = os.path.join(VALIDATE_FOLDER, "noisy")

        all_noisy_files = glob.glob(f"{NOISY_VAL}/**/*.npy")
        all_clean_files = glob.glob(f"{CLEAN_VAL}/**/*.npy")

        val_iter = data_generator(
            all_clean_files, all_noisy_files, batch_size=cnf.val_batch_size
        )

        preds_metrics = []
        for clean, noisy in val_iter:

            y_pred = model.predict(noisy)
            true = noisy - clean

            metric = mean_squared_error(
                true.reshape(cnf.sound_threshold, 80),
                y_pred.reshape(cnf.sound_threshold, 80),
            )

            preds_metrics.append(metric)

        print("Average MSE on val data:", np.mean(preds_metrics))
