# Training parameters
batch_size = 32
epochs = 1
weights_path = "weights/model_unet.h5"
save_weights = False
validation_steps = 16
steps_per_epoch = 3
train_proportion = 90

# Flag for cheking model on val data
CHECK_ON_VAL_DATA = True
val_batch_size = 1

# Threshold for every sound
sound_threshold = 864

# Paths
train_folder = "train"
val_folder = "val"
