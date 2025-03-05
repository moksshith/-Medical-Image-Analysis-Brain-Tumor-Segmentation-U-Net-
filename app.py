from flask import Flask, request, jsonify, send_file, render_template
import onnxruntime
import numpy as np
import cv2
import albumentations as A
from PIL import Image
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the ONNX model
model_onnx = "C:/Users/moksh/segmentation/brain-mri-unet.onnx"
session = onnxruntime.InferenceSession(model_onnx, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (np.array(image).astype(np.float32))/255.
    
    test_transform = A.Compose([
        A.Resize(width=128, height=128, p=1.0)
    ])
    
    aug = test_transform(image=image)
    image = aug['image']
    
    image = image.transpose((2,0,1))
    
    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])
    for i in range(image.shape[0]):
        image[i, :, :] = (image[i, :, :] - mean_vec[i]) / (std_vec[i])
    
    return image

def overlay_mask(image, mask, alpha=0.5):
    mask = np.squeeze(mask)
    masked = np.ma.masked_where(mask == 0, mask)
    
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    plt.imshow(masked, alpha=alpha, cmap='jet')
    plt.title('Original Image with Segmentation Overlay')
    plt.axis('off')
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close()
    
    return img_buf

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    preprocessed_image = preprocess_image(image)
    input_img = np.expand_dims(preprocessed_image, axis=0)
    
    result = session.run([output_name], {input_name: input_img})
    
    arr = np.array(result).squeeze()
    arr = (arr > 0.5).astype(np.uint8)
    
    resized_mask = cv2.resize(arr, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    overlay_img_buf = overlay_mask(original_image, resized_mask)
    
    # Convert original image to buffer
    original_img_buf = io.BytesIO()
    original_image_pil = Image.fromarray(original_image)
    original_image_pil.save(original_img_buf, format='PNG')
    original_img_buf.seek(0)
    
    # Convert segmentation mask to buffer
    mask_img_buf = io.BytesIO()
    mask_image_pil = Image.fromarray(resized_mask * 255)
    mask_image_pil.save(mask_img_buf, format='PNG')
    mask_img_buf.seek(0)
    
    return jsonify({
        'original_image': 'data:image/png;base64,' + base64.b64encode(original_img_buf.read()).decode('utf-8'),
        'segmented_image': 'data:image/png;base64,' + base64.b64encode(overlay_img_buf.read()).decode('utf-8'),
        'mask_image': 'data:image/png;base64,' + base64.b64encode(mask_img_buf.read()).decode('utf-8')
    })

if __name__ == '__main__':
    app.run(debug=True)