"""
USAGE:
python dataset_viewer_app.py \
  --images_dir data \
  --coco_json data/_annotations.coco.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk


# ----------------------------------------------------------
# COCO helpers
# ----------------------------------------------------------

def decode_mask(ann, img_h, img_w, resize_to_image=True):
    """
    Decode COCO segmentation to a binary mask.

    - For RLE: decodes at the annotation's native size (rle['size']).
    - For polygons: draws in the given img_h, img_w coordinate space.
    - If resize_to_image=True with RLE, we resize to (img_h, img_w) later,
      otherwise we keep the decoded size.
    """
    seg = ann["segmentation"]

    # --- RLE case ---
    if isinstance(seg, dict) and "counts" in seg:
        rle = seg
        if isinstance(rle["counts"], list):
            # rle["size"] is [h, w]
            rle = mask_utils.frPyObjects(rle, rle["size"][0], rle["size"][1])
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.uint8)

    # --- Polygon case ---
    elif isinstance(seg, list) and len(seg) > 0:
        # Assume polygons are already in the image coordinate space
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for poly in seg:
            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    else:
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

    h, w = mask.shape[:2]
    if resize_to_image and (h, w) != (img_h, img_w):
        print(f"[WARN] Resizing mask from {h}x{w} to match image {img_h}x{img_w}")
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    return mask


def rotate_array(arr, angle_deg):
    """Rotate an image/array by 0/90/180/270/360 degrees."""
    angle = int(angle_deg) % 360
    if angle == 0 or angle == 360:
        return arr
    if angle == 90:
        return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(arr, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Fallback
    return arr


def build_color_mask(img_shape, anns, cat_id_to_name, rotation_deg=0):
    """
    Build an RGB mask image (same size as original image) where:
      - induration -> red
      - sticker    -> green
      - others     -> blue

    For RLE masks, we:
      - decode at native annotation size,
      - rotate by rotation_deg,
      - then resize to match the image for visualization.
    """
    img_h, img_w = img_shape[:2]
    color_mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    for ann in anns:
        cat_id = ann["category_id"]
        name = cat_id_to_name.get(cat_id, str(cat_id)).lower()

        # Get raw mask in the annotation's native space (for RLE), no resize yet.
        raw_mask = decode_mask(ann, img_h, img_w, resize_to_image=False)

        # Rotate the mask in its own coordinate system
        rotated_mask = rotate_array(raw_mask, rotation_deg)

        # Now resize to image size for overlay if needed
        mh, mw = rotated_mask.shape[:2]
        if (mh, mw) != (img_h, img_w):
            rotated_mask = cv2.resize(
                rotated_mask.astype(np.uint8),
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST,
            )

        mask = (rotated_mask > 0).astype(np.uint8)

        if name == "induration":
            color = np.array([255, 0, 0], dtype=np.uint8)  # red
        elif name == "sticker":
            color = np.array([0, 255, 0], dtype=np.uint8)  # green
        else:
            color = np.array([0, 0, 255], dtype=np.uint8)  # blue

        color_mask[mask > 0] = color

    return color_mask


# ----------------------------------------------------------
# GUI App
# ----------------------------------------------------------

class TBViewerApp:
    def __init__(self, root, images_dir, coco):
        self.root = root
        self.images_dir = Path(images_dir)
        self.coco = coco

        self.root.geometry("1200x900")

        # Build indices
        self.images = {img["id"]: img for img in coco["images"]}
        self.anns_by_img = {img_id: [] for img_id in self.images.keys()}
        for ann in coco["annotations"]:
            self.anns_by_img[ann["image_id"]].append(ann)

        self.cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}

        # Collect pairs so we can sort reliably
        pairs = []
        for img_id, info in self.images.items():
            file_name = info["file_name"]
            path = self.images_dir / file_name
            if path.is_file():
                pairs.append((file_name, img_id))

        # Sort alphabetically by filename
        pairs.sort(key=lambda x: x[0].lower())

        # Unzip back into name list and id list
        self.display_names = [p[0] for p in pairs]
        self.valid_img_ids = [p[1] for p in pairs]

        if not self.valid_img_ids:
            raise RuntimeError("No valid images with matching files found.")

        # Store current image data
        self.current_img_id = self.valid_img_ids[0]
        self.current_img_rgb = None           # original image (no rotation)
        self.current_mask_rgb = None          # color mask, rotated per dropdown
        self.current_display_img = None       # PIL.Image
        self.current_photo = None             # ImageTk.PhotoImage
        self.scale_factor = 1.0

        # UI variables
        self.show_image = tk.BooleanVar(value=True)
        self.show_mask = tk.BooleanVar(value=True)
        self.rotation_angle = tk.StringVar(value="0")  # dropdown: 0,90,180,270,360

        self._build_ui()
        self._load_image(self.current_img_id)
        self._update_display()

    def _build_ui(self):
        self.root.title("TB Induration Viewer (mask rotation)")

        # Top frame: controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(control_frame, text="Image:").pack(side=tk.LEFT, padx=5)

        self.combo = ttk.Combobox(
            control_frame,
            values=self.display_names,
            state="readonly",
            width=50,
        )
        self.combo.current(0)
        self.combo.pack(side=tk.LEFT, padx=5)
        self.combo.bind("<<ComboboxSelected>>", self.on_image_change)

        # Checkbuttons for image/mask toggles
        self.chk_image = tk.Checkbutton(
            control_frame, text="Show Image", variable=self.show_image,
            command=self._update_display
        )
        self.chk_image.pack(side=tk.LEFT, padx=5)

        self.chk_mask = tk.Checkbutton(
            control_frame, text="Show Mask", variable=self.show_mask,
            command=self._update_display
        )
        self.chk_mask.pack(side=tk.LEFT, padx=5)

        # Rotation dropdown for MASK only
        tk.Label(control_frame, text="Mask rotation:").pack(side=tk.LEFT, padx=5)
        self.rotation_combo = ttk.Combobox(
            control_frame,
            values=["0", "90", "180", "270"],
            state="readonly",
            width=5,
            textvariable=self.rotation_angle,
        )
        self.rotation_combo.current(0)
        self.rotation_combo.pack(side=tk.LEFT, padx=2)
        self.rotation_combo.bind("<<ComboboxSelected>>", self.on_rotation_change)

        # Coordinate label
        self.coord_label = tk.Label(control_frame, text="x=?, y=?")
        self.coord_label.pack(side=tk.RIGHT, padx=10)

        # Keyboard navigation: arrow keys move between images
        self.root.bind("<Left>", self._show_prev_image)
        self.root.bind("<Right>", self._show_next_image)

        # Image display area
        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.image_label.bind("<Motion>", self.on_mouse_move)

    def on_image_change(self, event=None):
        idx = self.combo.current()
        self.current_img_id = self.valid_img_ids[idx]
        self._load_image(self.current_img_id)
        self._update_display()

    def on_rotation_change(self, event=None):
        # Just recompute the mask with new rotation
        self._load_mask_only()
        self._update_display()

    def _load_image(self, img_id):
        info = self.images[img_id]
        file_name = info["file_name"]
        path = self.images_dir / file_name

        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            print(f"[WARN] Could not read image: {path}")
            self.current_img_rgb = None
            self.current_mask_rgb = None
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.current_img_rgb = img_rgb

        # Load mask with current rotation
        self._load_mask_only()

    def _load_mask_only(self):
        if self.current_img_rgb is None:
            self.current_mask_rgb = None
            return

        rotation_deg = int(self.rotation_angle.get())
        img_rgb = self.current_img_rgb

        anns = self.anns_by_img.get(self.current_img_id, [])
        self.current_mask_rgb = build_color_mask(
            img_rgb.shape,
            anns,
            self.cat_id_to_name,
            rotation_deg=rotation_deg,
        )

    def _compose_display_image(self):
        """
        Combine image and mask according to toggles:
          - both off: black
          - image only
          - mask only
          - overlay
        Also computes scaling to fit window (max dimension).
        """
        if self.current_img_rgb is None:
            # Black placeholder
            disp = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            img = self.current_img_rgb
            mask = self.current_mask_rgb
            h, w = img.shape[:2]

            show_img = self.show_image.get()
            show_msk = self.show_mask.get()

            if not show_img and not show_msk:
                disp = np.zeros_like(img)
            elif show_img and not show_msk:
                disp = img.copy()
            elif not show_img and show_msk:
                # Mask on black background
                disp = mask.copy() if mask is not None else np.zeros_like(img)
            else:
                # Overlay
                if mask is None:
                    disp = img.copy()
                else:
                    disp = cv2.addWeighted(img, 1.0, mask, 0.4, 0)

        # Scale to fit a reasonable window (e.g., max 900 px in any dimension)
        max_dim = 900
        h, w = disp.shape[:2]
        scale = min(max_dim / h, max_dim / w, 1.0)
        self.scale_factor = scale

        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            disp_resized = cv2.resize(disp, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            disp_resized = disp

        # Convert to PIL Image
        pil_img = Image.fromarray(disp_resized)
        self.current_display_img = pil_img
        self.current_photo = ImageTk.PhotoImage(pil_img)

    def _update_display(self):
        self._compose_display_image()
        if self.current_photo is not None:
            self.image_label.configure(image=self.current_photo)

    def on_mouse_move(self, event):
        """
        Show image coordinates under mouse, mapped back to original image space.
        """
        if self.current_img_rgb is None:
            self.coord_label.config(text="x=?, y=?")
            return

        # event.x, event.y are in display coordinates
        x_disp, y_disp = event.x, event.y
        scale = self.scale_factor if self.scale_factor > 0 else 1.0

        x_img = int(x_disp / scale)
        y_img = int(y_disp / scale)

        h, w = self.current_img_rgb.shape[:2]
        if 0 <= x_img < w and 0 <= y_img < h:
            self.coord_label.config(text=f"x={x_img}, y={y_img}")
        else:
            self.coord_label.config(text="x=?, y=?")

    def _show_prev_image(self, event=None):
        """Select the previous image in the dropdown (if any)."""
        try:
            current_idx = self.combo.current()
        except tk.TclError:
            return

        if current_idx <= 0:
            return

        self.combo.current(current_idx - 1)
        self.on_image_change()

    def _show_next_image(self, event=None):
        """Select the next image in the dropdown (if any)."""
        try:
            current_idx = self.combo.current()
        except tk.TclError:
            return

        if current_idx >= len(self.display_names) - 1:
            return

        self.combo.current(current_idx + 1)
        self.on_image_change()

# ----------------------------------------------------------
# Main entry
# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Small interactive TB induration viewer with image/mask toggles and mask rotation."
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Directory with images (e.g., tb_data/train)",
    )
    parser.add_argument(
        "--coco_json",
        required=True,
        help="COCO JSON file (e.g., tb_data/train/_annotations.coco.json)",
    )
    args = parser.parse_args()

    with open(args.coco_json, "r") as f:
        coco = json.load(f)

    root = tk.Tk()
    app = TBViewerApp(root, args.images_dir, coco)
    root.mainloop()


if __name__ == "__main__":
    main()
