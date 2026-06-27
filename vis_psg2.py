import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# SETUP PATHS & REPO DICTIONARIES
# ---------------------------------------------------------
JSON_PATH = "data/psg/processed/psg_tra.json"
IMAGE_DIR = "data/coco/"          # Path to raw COCO/VG images
PANSEG_DIR = "data/coco/" # Path to panoptic PNG masks


# Ground truth predicate list defined by the OpenPSG/Visual Genome ontology
PREDICATES = [
    "over", "in front of", "beside", "on", "in", "attached to", "hanging from",
    "holding", "sitting on", "wears", "riding", "carrying", "eating", "looking at",
    "hitting", "kicking", "stepping on", "standing on", "leaning on", "wearing",
    "catching", "driving", "playing", "using", "watching", "inside", "under",
    "behind", "next to", "near", "has", "part of", "flying in", "covered in",
    "climbing", "lying on", "crossing", "standing in", "getting into", "holding hands with",
    "kissing", "hugging", "walking on", "running on", "skating on", "surfing on",
    "skateboarding on", "riding on", "standing behind", "sitting behind", "driving on",
    "walking in", "running in", "blocks", "painted on", "covered by"
]

# 1. Load Custom JSON
with open(JSON_PATH, "r") as f:
    psg_data = json.load(f)

# Index categories for quick string lookups
category_map = {cat["id"]: cat["name"] for cat in psg_data["categories"]}
image_map = {img["id"]: img for img in psg_data["images"]}

# Find an annotation entry that contains relationship edges
target_ann = None
for ann in psg_data["annotations"]:
    if len(ann.get("relations", [])) > 0:
        target_ann = ann
        break

if not target_ann:
    raise ValueError("No valid annotations with scene graph relations found.")

# 2. Extract Reference Files
image_meta = image_map[target_ann["image_id"]]
img_path = os.path.join(IMAGE_DIR, image_meta["file_name"])
mask_path = os.path.join(PANSEG_DIR, target_ann["file_name"])

# Load Images
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pan_mask_rgb = cv2.imread(mask_path)
pan_mask_rgb = cv2.cvtColor(pan_mask_rgb, cv2.COLOR_BGR2RGB)

# Convert RGB panoptic mask canvas back into true segment IDs
pan_id_map = (
    pan_mask_rgb[:, :, 0].astype(np.int32) +
    pan_mask_rgb[:, :, 1].astype(np.int32) * 256 +
    pan_mask_rgb[:, :, 2].astype(np.int32) * 256 * 256
)

# 3. Parse the first relation edge
segments = target_ann["segments_info"]
s_idx, o_idx, pred_id = target_ann["relations"][0]

sub_seg = segments[s_idx]
obj_seg = segments[o_idx]

sub_label = category_map[sub_seg["category_id"]]
obj_label = category_map[obj_seg["category_id"]]
rel_label = PREDICATES[pred_id] if pred_id < len(PREDICATES) else f"pred_{pred_id}"

print(f"Plotting relation: [{sub_label} (ID:{sub_seg['id']})] -> {rel_label} -> [{obj_label} (ID:{obj_seg['id']})]")

# 4. Filter Binary Segment Masks
sub_mask = (pan_id_map == sub_seg["id"])
obj_mask = (pan_id_map == obj_seg["id"])

def compute_centroid(mask):
    y, x = np.where(mask)
    if len(x) == 0 or len(y) == 0:
        return None
    return int(np.mean(x)), int(np.mean(y))

sub_center = compute_centroid(sub_mask)
obj_center = compute_centroid(obj_mask)

# 5. Render Scene Graph Diagram
plt.figure(figsize=(10, 8))
canvas = img.copy()

# Add color highlights to target masks
canvas[sub_mask] = canvas[sub_mask] * 0.5 + np.array([255, 0, 0]) * 0.5  # Red tint for Subject
canvas[obj_mask] = canvas[obj_mask] * 0.5 + np.array([0, 0, 255]) * 0.5  # Blue tint for Object

plt.imshow(canvas)
ax = plt.gca()

if sub_center and obj_center:
    # Directed Arrow
    ax.annotate(
        "", 
        xy=obj_center, 
        xytext=sub_center,
        arrowprops=dict(arrowstyle="-|>", color="lime", lw=3, mutation_scale=25)
    )
    # Edge Label
    midpoint = ((sub_center[0] + obj_center[0]) // 2, (sub_center[1] + obj_center[1]) // 2)
    ax.text(
        midpoint[0], midpoint[1], rel_label, 
        color="black", weight="bold", fontsize=11,
        bbox=dict(facecolor="lime", alpha=0.9, edgecolor="none", boxstyle="round,pad=0.3")
    )

# Label Vertices
if sub_center:
    ax.text(sub_center[0], sub_center[1], sub_label, color="white", weight="bold",
            bbox=dict(facecolor="red", alpha=0.8, boxstyle="round"))
if obj_center:
    ax.text(obj_center[0], obj_center[1], obj_label, color="white", weight="bold",
            bbox=dict(facecolor="blue", alpha=0.8, boxstyle="round"))

plt.title(f"Image ID: {target_ann['image_id']} Relationship Visualizer")
plt.axis("off")
plt.show()
# ... [Keep your existing mask processing & plotting setup from above] ...

plt.title(f"Image ID: {target_ann['image_id']} Relationship Visualizer")
plt.axis("off")

# 1. Clear extra margins around the plot box boundaries
plt.tight_layout()

# 2. Save the figure as a high-resolution PNG file
output_filename = f"psg_vis_{target_ann['image_id']}.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Successfully rendered and saved scene graph plot to: {output_filename}")

# 3. Clean up the internal figure memory buffer
plt.close()
