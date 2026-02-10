import tensorflow as tf
import tensorflow_datasets as tfds

gpu = tf.config.list_physical_devices("GPU")

if gpu:
    print("GPU detected so training on GPU")
else:
    print("GPU not detected so training on CPU")

(ds_train, ds_val) = tfds.load("cats_vs_dogs", split = ["train[:80%]", "train[80%:]"], as_supervised = True)

IMG_SIZE = 224
BATCH_SIZE = 32  

def preprocessing(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

ds_train = ds_train.map(preprocessing).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocessing).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

data_augmentation = tf.keras.models.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

base_model = tf.keras.applications.MobileNetV2(input_shape = (IMG_SIZE, IMG_SIZE, 3), include_top = False, weights = "imagenet")

input = tf.keras.Input(shape = (IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(input)
x = base_model(x, training = False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(input, output)

# Training Top Dense Layer
print("Top Dense Layer Training Starting now!")

base_model.trainable = False
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(ds_train, validation_data = ds_val, epochs = 10)

#Fine Tuning MobileNetV2 Layers
print("Fine Tuning starting now!")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

input = tf.keras.Input(shape = (IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(input)
x = base_model(x, training = True)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(input, output)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5), loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(ds_train, validation_data = ds_val, epochs = 5)

val_loss, val_acc = model.evaluate(ds_val)
print(f"Validation Accuracy: {(val_acc * 100):.2f}%")
print("Validation Loss: ", val_loss)

model.save("cats_vs_dogs_image_classifier_model.keras")

print("Your model has been trained and saved!")