import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import requests
import gradio as gr
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = torchvision.models.ResNet18_Weights.DEFAULT
model = torchvision.models.resnet18(weights=weights)
model.eval()
model.to(device)

labels_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
labels = requests.get(labels_url).json()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_top3_predictions(img):
    img = img.convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, 3)

    results = []
    for prob, idx in zip(top_probs, top_idxs):
        label = labels[str(idx.item())][1]
        results.append(f"{label}: {prob.item():.4f}")

    return "\n".join(results)

def generate_gradcam(img):
    img = img.convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[1].conv2
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0]
    acts = activations[0]

    weights_cam = grads.mean(dim=[2, 3], keepdim=True)
    cam = (weights_cam * acts).sum(dim=1)
    cam = torch.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    heatmap = cv2.resize(cam, (img.size[0], img.size[1]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    img_np = np.array(img)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

    pred_label = labels[str(pred_class.item())][1]

    forward_handle.remove()
    backward_handle.remove()

    return Image.fromarray(overlay), pred_label

def gradcam_app(image):
    if image is None:
        return None, "No image uploaded.", ""

    overlay, pred_label = generate_gradcam(image)
    top3 = get_top3_predictions(image)

    return overlay, pred_label, top3

demo = gr.Interface(
    fn=gradcam_app,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Image(type="pil", label="Grad-CAM Overlay"),
        gr.Textbox(label="Predicted Class"),
        gr.Textbox(label="Top-3 Predictions")
    ],
    title="Vision Model Interpretability with Grad-CAM",
    description="Upload an image to see a ResNet-18 prediction, top-3 classes, and a Grad-CAM heatmap."
)

if __name__ == "__main__":
    demo.launch()