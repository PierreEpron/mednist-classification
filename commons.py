
import os
import io
import torchvision as tv
from PIL import Image
import onnx
from skimage.transform import resize
import numpy as np

def get_model() :
    # Preprocessing: load the ONNX model
    model_path = os.path.join('models', 'model_opti.onnx')
    model = onnx.load(model_path)
    # Check the model
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print('The model is invalid: %s' % e)
    else:
        print('The model is valid!')
    return model

def transform_image(image_bytes) : 
    img = Image.open(io.BytesIO(image_bytes))
    a = np.asarray(img)
    if a.shape[0] != 64 or a.shape[1] != 64:
        img = Image.fromarray(resize(a, (64, 64), preserve_range=True))
    img_y = scaleImage(img)
    img_y.unsqueeze_(0)
    return img_y

def format_class_name(class_name):
    class_name = class_name.title()
    return class_name

# Pass a PIL image, return a tensor
def scaleImage(x):          
    toTensor = tv.transforms.ToTensor()
    y = toTensor(x)
    if(y.min() < y.max()):  
        y = (y - y.min())/(y.max() - y.min()) 
    z = y - y.mean()        
    return z

