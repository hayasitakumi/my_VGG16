from keras.applications.vgg16 import VGG16,decode_predictions
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

def vgg16():
  
  model = VGG16(weights='imagenet', include_top=True)

  img_path = 'elephant.jpg'
  img = image.load_img(img_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  pred = model.predict(x)
  results = decode_predictions(pred, top=3)[0]
  return results

if __name__ == "__main__":
  results = vgg16()
  for result in results:
    print(result)