# Vision Model Interpretability with Grad-CAM

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-yellow)](https://huggingface.co/spaces/dschechter27/vision_model_interpretability)

Interactive demo for **explaining CNN image classification** using **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

This project allows users to:

- Upload an image
- Classify it using a **ResNet-18 convolutional neural network**
- View the **top-3 predicted ImageNet classes**
- Visualize a **Grad-CAM heatmap** highlighting which regions influenced the model’s prediction

The demo is deployed as an **interactive web application using Gradio** and hosted on **Hugging Face Spaces**.

---

# Live Demo

Try the interactive application here:

https://huggingface.co/spaces/dschechter27/vision_model_interpretability

Upload an image and instantly see the model’s prediction and visual explanation.

---

# Example Results

### Cruise Ship Example  
Model prediction: **liner**

![Ship Example](assets/ship_example.png)

Grad-CAM highlights the **structure of the ship**, showing that the model focuses on the correct object region.

---

### Dog Example  
Model prediction: **Bernese Mountain Dog**

![Dog Example](assets/dog_example.png)

The heatmap concentrates on the **dog’s body and head**, demonstrating that the model uses meaningful visual features when making its prediction.

---

### Sports Car Example  
Model prediction: **sports car**

![Car Example](assets/car_example.png)

Grad-CAM focuses on the **car body and wheels**, confirming that the model is identifying relevant vehicle features.

---

# How Grad-CAM Works

Grad-CAM explains CNN predictions by analyzing the **gradient of the target class score with respect to convolutional feature maps**.

The importance weight for each feature map is computed as:

α_k = average( ∂y / ∂A_k )

Where:

- **A_k** = feature map *k* from the final convolutional layer  
- **y** = score for the predicted class  

The Grad-CAM heatmap is then computed as:

L_gradcam = ReLU( Σ_k α_k · A_k )

This produces a visualization highlighting **which regions of the image most influenced the model’s prediction**.

---
## Model Pipeline

```mermaid
flowchart LR

A[Upload Image] --> B[Image Preprocessing]
B --> C[ResNet-18 CNN]
C --> D[Class Prediction]

D --> E[Compute Gradients]
E --> F[Grad-CAM]
F --> G[Heatmap Generation]

G --> H[Overlay Heatmap on Image]
H --> I[Visual Explanation]
```
---
# Project Pipeline

Image  
→ ResNet-18 Prediction  
→ Gradient Backpropagation  
→ Grad-CAM Heatmap  
→ Visual Explanation

---

# Repository Structure

vision-model-interpretability/

assets/                 # README example screenshots  
images/                 # original input images  
results/                # Grad-CAM outputs  

app.py                  # Gradio web application  
requirements.txt        # project dependencies  
vision_model_interpretability.ipynb   # development notebook  
README.md  

---

# Technologies Used

- **PyTorch**
- **Torchvision**
- **Grad-CAM**
- **Gradio**
- **OpenCV**
- **NumPy**
- **Hugging Face Spaces**

---

# Running Locally

Install dependencies:

pip install -r requirements.txt

Run the web app:

python app.py

Then open the Gradio interface in your browser to upload images and visualize model explanations.

---

# Author

**David Schechter**  
MIT Class of 2030  

Interested in **machine learning, AI interpretability, and computer vision**.

GitHub: https://github.com/dschechter27

---

# License

MIT License
