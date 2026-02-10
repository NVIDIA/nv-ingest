import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from nemotron_page_elements_v3.model import define_model
from nemotron_page_elements_v3.utils import plot_sample, postprocess_preds_page_element, reformat_for_plotting

# Load image
path = "./example.png"
img = Image.open(path).convert("RGB")
img = np.array(img)

# Load model
model = define_model("page_element_v3")

# Inference
with torch.inference_mode():
    x = model.preprocess(img)
    preds = model(x, img.shape)[0]

print(preds)

# Post-processing
boxes, labels, scores = postprocess_preds_page_element(preds, model.thresholds_per_class, model.labels)

# Plot
boxes_plot, confs = reformat_for_plotting(boxes, labels, scores, img.shape, model.num_classes)

plt.figure(figsize=(15, 10))
plot_sample(img, boxes_plot, confs, labels=model.labels)
plt.show()