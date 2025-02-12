import os
import cv2
import pytesseract
import pandas as pd
import difflib
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import streamlit as st

# Configure logging
logging.basicConfig(filename='ocr_process.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
GROUND_TRUTH_CSV = "ground_truth.csv"
EXTRACTED_TEXT_CSV = "output.csv"
EVALUATION_CSV = "evaluation_results.csv"
RAW_IMAGE_DIR = "raw_images"

# Generate ground truth if not present
def generate_ground_truth():
    if not os.path.exists(GROUND_TRUTH_CSV):
        logging.info("Generating ground truth data...")
        data = []
        for filename in os.listdir(RAW_IMAGE_DIR):
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(RAW_IMAGE_DIR, filename)
                image = cv2.imread(image_path)
                extracted_text = pytesseract.image_to_string(image).strip()
                data.append([filename, extracted_text])
        df = pd.DataFrame(data, columns=["Filename", "Extracted Text"])
        df.to_csv(GROUND_TRUTH_CSV, index=False)
        logging.info("Ground truth generated successfully.")
    else:
        logging.info("Ground truth file already exists.")

# Load extracted text and ground truth data
def load_csv_data(file_path):
    data = {}
    try:
        df = pd.read_csv(file_path, dtype=str).fillna("")
        if "Filename" not in df.columns or "Extracted Text" not in df.columns:
            logging.error(f"Error: {file_path} does not have the required columns.")
            return {}
        for _, row in df.iterrows():
            filename = row["Filename"].strip().lower()
            extracted_text = str(row["Extracted Text"]).strip()
            data[filename] = extracted_text
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
    return data

# Evaluate OCR Performance
def normalize_text(text):
    return " ".join(str(text).lower().strip().split())

def calculate_similarity(text1, text2):
    return difflib.SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()

def main():
    generate_ground_truth()
    ground_truth_data = load_csv_data(GROUND_TRUTH_CSV)
    extracted_data = load_csv_data(EXTRACTED_TEXT_CSV)

    # Identify missing files and process them
    missing_files = set(ground_truth_data.keys()) - set(extracted_data.keys())
    if missing_files:
        logging.warning(f"Missing files: {missing_files}")
        for filename in missing_files:
            image_path = os.path.join(RAW_IMAGE_DIR, filename)
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                extracted_text = pytesseract.image_to_string(image).strip()
                extracted_data[filename] = extracted_text
            else:
                logging.error(f"Image {filename} not found.")

    # Save updated extracted data
    output_df = pd.DataFrame(list(extracted_data.items()), columns=["Filename", "Extracted Text"])
    output_df.to_csv(EXTRACTED_TEXT_CSV, index=False)

    # Perform evaluation
    results, similarity_scores, filenames = [], [], []
    for filename, true_text in ground_truth_data.items():
        extracted_text = extracted_data.get(filename, "")
        similarity_score = calculate_similarity(true_text, extracted_text)
        results.append([filename, true_text, extracted_text, similarity_score])
        similarity_scores.append(similarity_score)
        filenames.append(filename)

    df_results = pd.DataFrame(results, columns=["Filename", "Ground Truth", "Extracted Text", "Similarity Score"])
    df_results.to_csv(EVALUATION_CSV, index=False)

    logging.info(f"Total Files Evaluated: {len(results)}")
    if similarity_scores:
        logging.info(f"Average Similarity Score: {sum(similarity_scores) / len(similarity_scores):.2f}")
    else:
        logging.info("No valid similarity scores computed.")

    # Visualization
    if similarity_scores:
        plt.figure(figsize=(10, 5))
        plt.barh(filenames, similarity_scores, color='skyblue')
        plt.xlabel("Similarity Score")
        plt.ylabel("Filename")
        plt.title("OCR Accuracy Evaluation")
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automate OCR Evaluation')
    parser.add_argument('--run', action='store_true', help='Run the OCR Evaluation Pipeline')
    args = parser.parse_args()
    if args.run:
        main()

    # Streamlit UI
    st.title("OCR Evaluation System")
    if st.button("Run OCR Evaluation"):
        main()
        st.success("OCR evaluation completed successfully!")
