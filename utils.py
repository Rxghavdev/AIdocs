import tensorflow as tf

def cleanImage(img, height=150, width=150):
    img = img.resize((height, width))
    img = img.convert('RGB')  # Convert to RGB mode
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)

    return img_array
