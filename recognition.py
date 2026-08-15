"""
Project 4: Image or Text Recognition (Basic)
DecodeLabs AI Internship — Combined Script
------------------------------------------------
Does BOTH:
  1. Image Recognition — classifies objects in a photo using a
     pre-trained MobileNetV2 model (ImageNet, 1000 classes).
  2. Text Recognition (OCR) — extracts text from an image using
     pytesseract.

Usage:
    python recognition.py --image path/to/photo.jpg
    python recognition.py --text path/to/text_image.png
    python recognition.py --image photo.jpg --text text_image.png
    python recognition.py            (runs both on the bundled sample images)
"""

import sys
import argparse
import numpy as np
from PIL import Image
import pytesseract
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image as keras_image

# If Tesseract isn't on your PATH (common on Windows), uncomment and set this:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def recognize_image(img_path, top_k=3):
    """Classify objects in an image using pre-trained MobileNetV2."""
    print("Loading pre-trained MobileNetV2 model (ImageNet weights)...")
    model = MobileNetV2(weights="imagenet")

    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    print(f"Running image recognition on: {img_path}")
    predictions = model.predict(img_array, verbose=0)
    results = decode_predictions(predictions, top=top_k)[0]

    print("\n=== Image Recognition Results ===")
    print(f"Image: {img_path}")
    print("-" * 40)
    for i, (_, label, confidence) in enumerate(results, start=1):
        print(f"{i}. {label.replace('_', ' ').title():<25} {confidence * 100:5.2f}%")
    print("-" * 40)

    return results


def recognize_text(img_path):
    """Extract text from an image using pytesseract OCR."""
    print(f"Running text recognition on: {img_path}")

    img = Image.open(img_path)
    gray_img = img.convert("L")
    extracted_text = pytesseract.image_to_string(gray_img)

    print("\n=== Text Recognition (OCR) Results ===")
    print(f"Image: {img_path}")
    print("-" * 40)
    cleaned = extracted_text.strip()
    print(cleaned if cleaned else "(No text detected)")
    print("-" * 40)

    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Basic image or text recognition using pre-trained models."
    )
    parser.add_argument("--image", help="Path to an image for object recognition")
    parser.add_argument("--text", help="Path to an image for text recognition (OCR)")
    args = parser.parse_args()

    print("=" * 50)
    print(" PROJECT 4: IMAGE OR TEXT RECOGNITION (BASIC)")
    print("=" * 50)

    # No arguments given -> run both on bundled sample images
    if not args.image and not args.text:
        args.image = "sample_images/sample_dog.jpg"
        args.text = "sample_images/sample_text.png"

    if args.image:
        print("\n\n[1] IMAGE RECOGNITION")
        recognize_image(args.image)

    if args.text:
        print("\n\n[2] TEXT RECOGNITION (OCR)")
        recognize_text(args.text)

    print("\nDone.")


if __name__ == "__main__":
    main()