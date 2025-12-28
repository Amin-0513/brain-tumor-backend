import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tumor_prediction import BrainTumorCNN
import warnings
import base64
warnings.filterwarnings('ignore')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the saved model
model = BrainTumorCNN().to(device)
model.load_state_dict(torch.load(r"E:\Brain tumor Detection\models\brain_tumor_cnn_model.pth", map_location=device))
model.eval()
print("Model loaded successfully!")

# Class names (must match training order)
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Function to predict and display image
def predict_and_display(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    
    # Display original image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Transform for model
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probabilities, 1)
        predicted_class = class_names[pred.item()]
        confidence_score = confidence.item()
    
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class}\\nConfidence: {confidence_score:.4f}")
    plt.axis('off')
    plt.tight_layout()
    plt.close()
    
    # Print all class probabilities
    print("\\nClass Probabilities:")
    for i, class_name in enumerate(class_names):
        prob = probabilities[0][i].item()
        print(f"{class_name}: {prob:.4f}")
    
    return predicted_class, confidence_score, image

# Function for LIME explanation
def explain_with_lime(image_path, top_labels=4, num_samples=1000):
    # Load image
    original_image = Image.open(image_path).convert("RGB")
    
    # Define prediction function for LIME
    def batch_predict(images):
        model.eval()
        batch = torch.stack([
            transform(Image.fromarray(image.astype('uint8'), 'RGB')) 
            for image in images
        ]).to(device)
        
        with torch.no_grad():
            outputs = model(batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()
    
    # Convert image to numpy array for LIME
    image_np = np.array(original_image)
    
    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    print("Generating LIME explanation... This may take a moment.")
    
    # Explain the prediction
    explanation = explainer.explain_instance(
        image_np, 
        batch_predict, 
        top_labels=top_labels, 
        hide_color=0, 
        num_samples=num_samples
    )
    
    return explanation, image_np
import io
import base64
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for servers
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy as np

def display_lime_explanation(explanation, image_np, predicted_class, class_names):
    """
    Converts LIME explanation to a base64-encoded PNG image.
    Args:
        explanation: LIME explanation object
        image_np: original image as numpy array
        predicted_class: predicted class name
        class_names: list of all class names

    Returns:
        img_base64: base64-encoded string of the explanation figure
    """
    # Get index of predicted class
    pred_idx = class_names.index(predicted_class)
    
    # Get explanation for the predicted class
    temp, mask = explanation.get_image_and_mask(
        pred_idx,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title(f"Original Image\nPredicted: {predicted_class}")
    axes[0].axis('off')
    
    # LIME explanation
    explanation_img = mark_boundaries(temp, mask)
    axes[1].imshow(explanation_img)
    axes[1].set_title("LIME Explanation\n(Green areas support prediction)")
    axes[1].axis('off')
    
    # Heatmap
    axes[2].imshow(mask, cmap='RdBu', alpha=0.7)
    axes[2].set_title("Importance Heatmap\n(Red: More important)")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Convert figure to Base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_base64

# Main execution
if __name__ == "__main__":
    # Test image path - replace with your image path
    test_image_path = r"E:\Brain tumor Detection\dataset\brain-tumor-mri-dataset\glioma\gl-0014.jpg"
    
    try:
        # Step 1: Make prediction and display results
        print("Step 1: Making prediction...")
        predicted_class, confidence, original_image = predict_and_display(test_image_path)
        
        # Step 2: Generate LIME explanation
        print("\\nStep 2: Generating LIME explanation...")
        explanation, image_np = explain_with_lime(test_image_path)
        
        # Step 3: Display LIME explanation
        print("\\nStep 3: Displaying LIME explanation...")
        base_image=display_lime_explanation(explanation, image_np, predicted_class, class_names)
        print(base_image)

        
        print(f"\\n✅ Analysis complete!")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
    except FileNotFoundError:
        print(f"Error: Image file not found at {test_image_path}")
        print("Please update the test_image_path variable with a valid image path.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed the required packages:")
        print("pip install lime scikit-image")