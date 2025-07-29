import sys
import os
from PIL import Image
import torch
from vit import load_datasets
from util import (
    show_vit_attention_on_image,
    load_embeddings_and_filenames,
    get_nearest_images,
    get_filenames_for_indices,
    generate_embeddings_and_metadata,
    get_cls_token
)
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.utils.data import Subset
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QComboBox, QFrame, QCheckBox, QGridLayout, QStackedLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# Load model and processor
model = ViTForImageClassification.from_pretrained('./temp/4-tennis', use_safetensors=True, attn_implementation="eager")
processor = ViTImageProcessor.from_pretrained('./temp/4-tennis')

train_dataset, val_dataset, classes = load_datasets()
full_dataset = train_dataset.dataset
all_indices = list(range(len(full_dataset)))
full_subset = Subset(full_dataset, all_indices)

if not os.path.exists("embeddings.npy") or not os.path.exists("metadata.npy"):
    generate_embeddings_and_metadata(model, full_subset, processor, "embeddings.npy", "metadata.npy")

embeddings, filenames = load_embeddings_and_filenames("embeddings.npy", "metadata.npy")

def pil2pixmap(im):
    if im.mode != "RGB":
        im = im.convert("RGB")
    data = im.tobytes("raw", "RGB")
    qimg = QImage(data, im.size[0], im.size[1], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def get_true_class_from_filename(fname):
    return os.path.basename(os.path.dirname(fname))

def get_predicted_class_and_confidence(model, processor, img, classes):
    model.eval()
    inputs = processor(img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs[0, pred].item()
    return classes[pred], confidence

class ImageResultWidget(QWidget):
    def __init__(self, orig_img, attn_img, true_class, pred_class, confidence, distance, show_attn=False, parent=None):
        super().__init__(parent)
        self.orig_img = orig_img
        self.attn_img = attn_img
        self.true_class = true_class
        self.pred_class = pred_class
        self.confidence = confidence
        self.distance = distance
        self.show_attn = show_attn
        self.init_ui()

    def init_ui(self):
        img_size = 225
        self.img_label = QLabel()
        self.img_label.setPixmap(pil2pixmap(self.orig_img).scaled(img_size, img_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.img_label.setFixedSize(img_size, img_size)
        self.img_label.setAlignment(Qt.AlignCenter)

        self.attn_label = QLabel()
        self.attn_label.setPixmap(pil2pixmap(self.attn_img).scaled(img_size, img_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.attn_label.setFixedSize(img_size, img_size)
        self.attn_label.setAlignment(Qt.AlignCenter)

        self.stack = QStackedLayout()
        self.stack.addWidget(self.img_label)
        self.stack.addWidget(self.attn_label)
        self.stack.setCurrentIndex(1 if self.show_attn else 0)

        self.class_label = QLabel(f"True: {self.true_class} | Pred: {self.pred_class}")
        self.class_label.setStyleSheet("color: #fff; font-size: 16px; margin-top: 2px; margin-bottom: 0px;")
        self.confidence_label = QLabel(f"Confidence: {self.confidence:.2%}")
        self.confidence_label.setStyleSheet("color: #aaa; font-size: 14px; margin-top: 0px; margin-bottom: 0px;")
        self.dist_label = QLabel(f"Distance: {self.distance:.4f}")
        self.dist_label.setStyleSheet("color: #aaa; font-size: 14px; margin-bottom: 8px; margin-top: 0px;")

        vbox = QVBoxLayout()
        vbox.setSpacing(2)  
        vbox.addLayout(self.stack)
        vbox.addWidget(self.class_label)
        vbox.addWidget(self.confidence_label)
        vbox.addWidget(self.dist_label)
        vbox.setAlignment(Qt.AlignHCenter)
        self.setLayout(vbox)
        self.setStyleSheet("background-color: #181818;")

    def set_overlay(self, show_attn):
        self.stack.setCurrentIndex(1 if show_attn else 0)

class ViTVectorbaseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎾ViT Vectorbase")
        self.setStyleSheet("background-color: #181818; color: #fff;")
        self.resize(700, 350)
        self.image_widgets = []
        self.init_ui()

    def init_ui(self):
        title = QLabel("🎾ViT Vectorbase")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #fff; margin-bottom: 20px;")

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.setStyleSheet("background-color: #333; color: #fff; padding: 10px; border-radius: 5px;")
        self.upload_btn.clicked.connect(self.upload_image)

        # Toggle for attention overlay
        self.overlay_toggle = QCheckBox("Show Attention Overlay for All Images")
        self.overlay_toggle.setStyleSheet("color: #fff; font-size: 16px; margin: 10px 0;")
        self.overlay_toggle.stateChanged.connect(self.toggle_all_overlays)

        # Dropdown for user to input true class
        self.true_class_dropdown = QComboBox()
        self.true_class_dropdown.addItems(classes)
        self.true_class_dropdown.setStyleSheet("background-color: #222; color: #fff; font-size: 16px;")
        self.true_class_label = QLabel("Select true class for uploaded image:")
        self.true_class_label.setStyleSheet("color: #fff; font-size: 16px; margin: 10px 0;")

        self.results_frame = QFrame()
        self.results_layout = QGridLayout()
        self.results_frame.setLayout(self.results_layout)
        self.results_frame.setStyleSheet("background-color: #181818;")

        vbox = QVBoxLayout()
        vbox.addWidget(title)
        vbox.addWidget(self.upload_btn)
        vbox.addWidget(self.overlay_toggle)
        vbox.addWidget(self.true_class_label)
        vbox.addWidget(self.true_class_dropdown)
        vbox.addWidget(self.results_frame)
        self.setLayout(vbox)

    def toggle_all_overlays(self, state):
        show_attn = state == Qt.Checked
        for widget in self.image_widgets:
            widget.set_overlay(show_attn)

    def toggle_true_class_input(self, state):
        self.true_class_dropdown.setVisible(state == Qt.Checked)

    def upload_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if fname:
            img = Image.open(fname).convert("RGB")
            query_embedding = get_cls_token(model, processor, img).squeeze(0).cpu().numpy()
            indices, distances = get_nearest_images(embeddings, query_embedding, n_neighbors=5)
            matched_filenames = get_filenames_for_indices(filenames, indices)

            for i in reversed(range(self.results_layout.count())):
                widget = self.results_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            self.image_widgets = []

            true_class = self.true_class_dropdown.currentText()
            pred_class, confidence = get_predicted_class_and_confidence(model, processor, img, classes)
            attn_img = show_vit_attention_on_image(model, processor, img)
            self.image_widgets.append(ImageResultWidget(img, attn_img, true_class, pred_class, confidence, 0.0, show_attn=self.overlay_toggle.isChecked()))
            self.results_layout.addWidget(self.image_widgets[-1], 0, 0)

            for i, (match_fname, dist) in enumerate(zip(matched_filenames, distances)):
                match_img = Image.open(match_fname).convert("RGB")
                match_true_class = get_true_class_from_filename(match_fname)
                match_pred_class, match_confidence = get_predicted_class_and_confidence(model, processor, match_img, classes)
                match_attn_img = show_vit_attention_on_image(model, processor, match_img)
                self.image_widgets.append(ImageResultWidget(match_img, match_attn_img, match_true_class, match_pred_class, match_confidence, dist, show_attn=self.overlay_toggle.isChecked()))
                if i < 2: 
                    row = 0
                    col = i + 1 
                else:
                    row = 1
                    col = i - 2 
                self.results_layout.addWidget(self.image_widgets[-1], row, col)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ViTVectorbaseApp()
    window.show()
    sys.exit(app.exec_())

