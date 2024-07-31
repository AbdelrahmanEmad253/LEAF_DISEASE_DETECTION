#LINK TO DATASET:      https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

# importing libraries
import tensorflow as tf

# data preprocessing
# training image preprocessing
training_set = tf.keras.utils.image_dataset_from_directory(
    'dataset/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

# validation image preprocessing
validation_set = tf.keras.utils.image_dataset_from_directory(
    'dataset/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


for x,y in training_set:
    print(x,x.shape)
    print(y,y.shape)
    break


# to avoid overshooting
# 1- choose small learning rate default 0.001 we are taking 0.0001
# 2- there may be chance of underfitting so increase number of neuron
# 3- add more convolution layer to extract more feature from images there may be possibility that model unable to capture relevant feature or model is confusing due to
# lack of feature so feed with more feature

# building model
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense, Conv2D, Dropout
from tensorflow.keras.models import Sequential

model = Sequential()


# buildng convolution layer
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[128, 128, 3]))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Dropout(0.25)) # to avoid overfitting

model.add(Flatten())
model.add(Dense(units=1500, activation='relu'))

# output layer
model.add(Dense(units=38, activation='softmax'))

model.add(Dropout(0.4))

# compiling model
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# model training
training_history = model.fit(x = training_set , validation_data = validation_set , epochs=10)


# model evaluation
train_loss, train_acc = model.evaluate(training_set)
print(train_loss, train_acc)


# model on validation set
val_loss, val_acc = model.evaluate(validation_set)
print(val_loss, val_acc)


# saving model
model.save('trained_model.h5')
model.save('trained_model.keras')
print(training_history.history)


# recording history in jason
import json
with open("training_history.json", "w")as f:
    json.dump(training_history.history, f)


# accuracy visualization
import matplotlib.pyplot as plt
epochs = [i for i in range(1,11)]
plt.plot(epochs,training_history.history['accuracy'],color = "red" , label="Training Accuracy")
plt.plot(epochs,training_history.history['val_accuracy'],color = "blue" , label="Validation Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy Result")
plt.title("Visualization of Accuracy vs No. of Epochs")
plt.legend()
plt.show()


# some other metrics for model evaluation
class_name = validation_set.class_names
test_set = tf.keras.utils.image_dataset_from_directory(
    'dataset/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


y_pred = model.predict(test_set)
print(y_pred,y_pred.shape)


predicted_categories = tf.argmax(y_pred,axis=1)
print(predicted_categories)


true_categories = tf.concat([y for x,y in test_set], axis=0)
print(true_categories)


y_true = tf.argmax(true_categories,axis=1)
print(y_true)

# classification_report, confusion_matrix visualization
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print(classification_report(y_true, predicted_categories, target_names=class_name))
cm = confusion_matrix(y_true, predicted_categories)
print(cm.shape)
plt.figure(figsize=(40,40))
sns.heatmap(cm, annot=True, annot_kws={"size":10})
plt.xlabel("Predicted Class", fontsize=40)
plt.ylabel("Actual Class", fontsize=40)
plt.title("Plant Disease Prediction Confusion Matrix", fontsize=50)
plt.show()
