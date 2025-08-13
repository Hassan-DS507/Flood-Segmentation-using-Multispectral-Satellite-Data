import os
import io
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file
import rasterio
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model definition
class WaterSegmenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Conv2d(12, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 3, kernel_size=1)
        )
        self.model = smp.Unet(
            encoder_name='resnet101',
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
    
    def forward(self, x):
        x = self.proj(x)
        return self.model(x)

# Initialize and load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WaterSegmenter().to(device)

try:
    state_dict = torch.load('DeepLabV3_model.pth', map_location=device)
    # Handle key naming differences
    state_dict = {k.replace('unet.', 'model.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully (some weights may not have been loaded)")
    
    # Test the model
    test_input = torch.randn(1, 12, 256, 256).to(device)
    with torch.no_grad():
        output = model(test_input)
    print(f"Model test successful! Output shape: {output.shape}")
    
except Exception as e:
    print(f"Error loading or testing model: {str(e)}")
    raise

model.eval()

def process_image(image_path):
    """Process the uploaded image and generate predictions"""
    try:
        # Read the 12-band image
        with rasterio.open(image_path) as src:
            bands = src.read()
        
        # Normalize and prepare the input
        bands = bands.astype(np.float32)
        bands = (bands - bands.min(axis=(1, 2), keepdims=True)) / (
            bands.max(axis=(1, 2), keepdims=True) - bands.min(axis=(1, 2), keepdims=True) + 1e-7)
        
        # Convert to tensor and predict
        input_tensor = torch.from_numpy(bands).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        
        # Get prediction mask
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        binary_mask = (pred_mask > 0.5).astype(np.uint8)
        
        # Create RGB visualization (using bands 4,3,2 for standard false color)
        rgb_image = np.stack([
            bands[3],  # Red
            bands[2],  # Green
            bands[1]   # Blue
        ], axis=-1)
        
        # Normalize RGB for display
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        
        # Create overlay
        overlay = rgb_image.copy()
        overlay[binary_mask == 1] = [0.0, 0.5, 1.0]  # Blue overlay
        
        # Convert to PIL Images
        rgb_img = Image.fromarray((rgb_image * 255).astype(np.uint8))
        mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        
        return rgb_img, mask_img, overlay_img
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.tif')
        file.save(filename)
        
        # Process the image
        rgb_img, mask_img, overlay_img = process_image(filename)
        
        # Save processed images
        rgb_path = os.path.join(app.config['UPLOAD_FOLDER'], 'rgb.png')
        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask.png')
        overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], 'overlay.png')
        
        rgb_img.save(rgb_path)
        mask_img.save(mask_path)
        overlay_img.save(overlay_path)
        
        return render_template('result.html', 
                             rgb_image='uploads/rgb.png',
                             mask_image='uploads/mask.png',
                             overlay_image='uploads/overlay.png')
    
    return redirect(url_for('index'))

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'tif', 'tiff'}

@app.route('/download_mask')
def download_mask():
    """Allow downloading the predicted mask"""
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask.png')
    return send_file(mask_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)