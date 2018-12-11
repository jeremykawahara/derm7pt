import numpy as np
from PIL import Image
from keras.preprocessing import image


def crop_resize_img(img_path, target_size, crop_amount):
    """Load, crop, and resize an image."""
    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = x[crop_amount:-crop_amount, crop_amount:-crop_amount, :]
    ximg = Image.fromarray(np.uint8(x))
    ximg_resize = ximg.resize((target_size[0], target_size[1]))
    x = image.img_to_array(ximg_resize)

    return x


def spatial_average_pooling(x):
    """Average across the entire spatial dimensions."""
    return np.squeeze(x).mean(axis=0).mean(axis=0)


def deep_features(img_paths, model, func_preprocess_input, target_size=(224, 224, 3),
                  crop_amount=None, flip_axis=None, func_postprocess_features=spatial_average_pooling):
    """Computes deep features for the images."""

    features = []
    for img_path in img_paths:
        if crop_amount is None:
            img = image.load_img(img_path, target_size=target_size)
            x = image.img_to_array(img)
        else:
            x = crop_resize_img(img_path, target_size, crop_amount)

        if flip_axis is not None:
            x = image.flip_axis(x, flip_axis)

        x = np.expand_dims(x, axis=0)
        x = func_preprocess_input(x)
        responses = model.predict(x)

        if func_postprocess_features is not None:
            responses = func_postprocess_features(responses)

        features.append(responses)

    features = np.squeeze(np.asarray(features))

    return features
