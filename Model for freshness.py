import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, SeparableConv2D, MaxPooling2D, Dropout, Flatten, Dense

# Load the pre-trained MobileNetV2 model
mobilenetv2_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Freeze the layers in the MobileNetV2 model
for layer in mobilenetv2_model.layers:
    layer.trainable = False

# Create a new Sequential model
model = Sequential()

# Add the MobileNetV2 model to the new model (up to the last convolutional layer)
model.add(mobilenetv2_model)

# Add the rest of the custom layers
model.add(BatchNormalization())
model.add(SeparableConv2D(64, (3, 3), depthwise_initializer='he_uniform', padding='same', activation='relu'))
model.add(SeparableConv2D(64, (3, 3), depthwise_initializer='he_uniform', padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), kernel_initializer='he_uniform', padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.build(input_shape=(None, 100, 100, 3))

# Print the summary of the model
model.summary()
