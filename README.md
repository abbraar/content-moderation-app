# Book Content Moderation System

A Streamlit application that uses Google's Gemini API to detect harmful content in both English and Arabic text from books.

## Features

- Supports both English and Arabic text analysis
- Accepts text input or file upload
- Detects harmful, inappropriate, or offensive content
- Provides confidence scores and categories for detected content
- Simple and intuitive web interface

## Prerequisites

- Python 3.7 or higher
- Google API key with access to the Gemini API

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## How to Run

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Usage

1. Either paste your text in the text area or upload a text file
2. Click the "Analyze Content" button
3. View the results showing if any harmful content was detected
4. Click "View Analysis Details" to see the full analysis

## Note

Make sure to keep your Google API key secure and never commit it to version control. The `.env` file is included in `.gitignore` by default.
