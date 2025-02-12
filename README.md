**Custom Object Character Recognition (OCR)**

**Overview**

This project builds a Custom OCR system by combining YOLO and Tesseract to extract specific content from lab reports and convert it into an editable file.

**Project Workflow**

**1. Set Up the Environment**

Configure the Tesseract notebook instance.

**2. Data Preparation**

Upload Dataset to S3: Stored the dataset .

Prepare the Dataset: Preprocess the dataset using Tesseract (resizing images, cleaning, etc.).

**3. Model Training**

Train YOLOv3 Model: Use Tesseract training jobs to train the YOLOv3 model on a custom dataset.

**4. Inference and Post-Processing**

Run Inference: Deploy YOLOv3 on Tesseract and store detected object coordinates.

Preprocess Detected Regions: Extract relevant text regions.

Extract Text Using Tesseract: Process preprocessed images through Tesseract using SageMaker.

**5. Evaluation and Optimization**

Evaluate OCR Performance: Compare the extracted text with ground truth for accuracy analysis.

Optimize the Workflow: Fine-tune the model and preprocessing steps.

**6. Automation and Deployment**

Automate the Workflow: Implement a pipeline to automate the entire process.

Deploy the Solution: Deploy on SageMaker for real-time OCR processing of lab reports.

**Features**

Custom object detection using YOLOv3

Text extraction with Tesseract OCR

Automated processing and real-time inference

Scalable deployment using AWS SageMaker



**Prepared by**: Vicky Saini
