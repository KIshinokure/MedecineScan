import cv2
import numpy as np
import torch
from PIL import Image

def run_grad_cam(model, input_tensor, target_category, original_image):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-3]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_forward_hook(backward_hook)

    model.zero_grad()
    input_tensor.requires_grad = True
    output = model(input_tensor)
    
    loss = output[0, target_category]
    loss.backward()

    grads = gradients[0].cpu().data.numpy()[0]
    f_maps = activations[0].cpu().data.numpy()[0]

    forward_handle.remove()
    backward_handle.remove()

    axes = tuple(range(1, gradients[0].ndim - 1)) if gradients[0].ndim > 2 else 0
    weights = np.mean(grads, axis=axes)
    
    cam = np.zeros(f_maps.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * f_maps[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (original_image.size[0], original_image.size[1]))
    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_np = np.array(original_image.convert("RGB"))
    result = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    return Image.fromarray(result)
