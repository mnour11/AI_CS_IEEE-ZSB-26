# Arabic OCR System - FastAPI Microservice

## Project Overview
This project is an end-to-end Arabic Optical Character Recognition (OCR) system. It features a Deep Learning model built using Convolutional Recurrent Neural Networks (CRNN) and Connectionist Temporal Classification (CTC) loss to recognize Arabic text from images. The trained model is deployed as a microservice using **FastAPI**.

## Project Structure
```text
arabic-ocr-system/
├── app/
│   ├── main.py                # FastAPI application and endpoints
│   └── model.py               # Model loading and image preprocessing logic
├── weights/
│   ├── best_ocr_model.keras   # Trained CRNN model weights
│   └── vocab.json             # Extracted vocabulary mapping
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Setup and Installation

Follow these steps to set up the project locally:

**1. Create a Virtual Environment**
It is recommended to use a virtual environment to manage dependencies:
```bash
python -m venv venv
```

**2. Activate the Virtual Environment**
*   **Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
*   **Linux/Mac:**
    ```bash
    source venv/bin/activate
    ```

**3. Install Dependencies**
Install the required libraries listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI server using Uvicorn:
```bash
uvicorn app.main:app --reload
```
The server will start running at `[http://127.0.0.1:8000](http://127.0.0.1:8000)`.

## API Endpoints & Usage

Once the server is running, you can access the interactive API documentation (Swagger UI) at:
**[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

### 1. Health Check
*   **Endpoint:** `GET /`
*   **Description:** Verifies that the API is up and running.
*   **Response:**
    ```json
    {"status": "System is running perfectly"}
    ```

### 2. Predict Text
*   **Endpoint:** `POST /predict`
*   **Description:** Accepts an image file (e.g., ID card crop) and returns the recognized Arabic text.
*   **Usage via Swagger UI:**
    1. Navigate to the `/predict` endpoint.
    2. Click **Try it out**.
    3. Upload an image file.
    4. Click **Execute**.
*   **Response Format:**
    ```json
    {
      "filename": "image_name.jpg",
      "predicted_text": "النص العربي المستخرج"
    }
    ```