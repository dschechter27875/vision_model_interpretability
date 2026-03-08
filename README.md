---
title: Vision Model Interpretability
emoji: 🔍
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Interactive Grad-CAM visualization
---

# Vision Model Interpretability with Grad-CAM  
**David Schechter**

Upload an image to:

- classify it using **ResNet-18**
- view the **top-3 predictions**
- visualize a **Grad-CAM heatmap** highlighting the image regions that influenced the model’s decision

This demo explores **model interpretability in computer vision** by showing how gradients from a convolutional neural network can be used to explain predictions.