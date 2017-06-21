import tensorflow.contrib.keras as keras


def pretrained(nc):
    # input_tensor = keras.layers.Input(batch_shape=(None, 64, 64, 3))
    # vgg = keras.applications.VGG16(input_tensor=input_tensor, input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
    vgg = keras.applications.VGG16(weights='imagenet', include_top=False)
    for layer in vgg.layers:
        layer.trainable = False
    x = vgg.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dense(2048, activation='relu')(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    predictions = keras.layers.Dense(nc, activation='sigmoid')(x)
    model = keras.models.Model(inputs=vgg.input, outputs=predictions)
    model.compile(keras.optimizers.SGD(lr=1e-4, momentum=0.9),
                  keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    return model


def unet(nc, h, w, c):
    inputs = keras.layers.Input((h, w, c))
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = keras.layers.concatenate([keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = keras.layers.concatenate([keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = keras.layers.concatenate([keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = keras.layers.concatenate([keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    flat = keras.layers.Flatten()(conv9)
    dense = keras.layers.Dense(nc, activation='sigmoid')(flat)

    model = keras.models.Model(inputs=[inputs], outputs=[dense])

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-5), loss=keras.losses.binary_crossentropy)

    return model


def multi_label_cnn(nc, h, w, c):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(
              filters=32,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid',
              input_shape=(h, w, c)))
    model.add(keras.layers.Conv2D(
              filters=32,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.Conv2D(
              filters=32,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv2D(
              filters=64,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.Conv2D(
              filters=64,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.Conv2D(
              filters=64,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv2D(
              filters=128,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.Conv2D(
              filters=128,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.Conv2D(
              filters=128,
              kernel_size=(3, 3),
              activation='relu',
              strides=1,
              padding='valid'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(nc, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam')
    return model
