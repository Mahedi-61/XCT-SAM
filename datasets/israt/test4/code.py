import numpy as np
import cv2
from scipy import ndimage as ndi
from PIL import Image
import matplotlib.pyplot as plt

# --- Load image ---
img_path = "./input/set1sample4raw_0016.png"
img = Image.open(img_path).convert("L")
img_arr = np.array(img)

# --- Threshold to detect metal region ---
threshold = 10  # adjust if needed
mask = img_arr > threshold

# --- Fill holes and smooth defects ---
mask_filled = ndi.binary_closing(mask, structure=np.ones((5, 5)))
mask_filled = ndi.binary_fill_holes(mask_filled)

# --- Fit circle around detected region ---
contours, _ = cv2.findContours(mask_filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)
(x, y), radius = cv2.minEnclosingCircle(cnt)

metal_mask = np.zeros_like(mask_filled, dtype=np.uint8)
cv2.circle(metal_mask, (int(x), int(y)), int(radius), 1, -1)
valid_region = metal_mask.astype(bool)

# Now valid_region is a clean circular mask of the metal area
plt.imshow(valid_region, cmap="gray")
plt.show()
