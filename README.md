# DermAssist-AI 🩺
**Imagine Cup 2026 Submission** *Accessible Skin Condition Classification Powered by Microsoft Azure*

## 🌟 Overview
DermAssist-AI is a healthcare solution designed for non-English speakers. It uses a local TensorFlow Lite model for diagnosis and integrates three Microsoft Azure services to ensure the results are accessible to everyone, regardless of language or literacy.

## 🚀 Azure AI Integration
- **Azure AI Vision:** Used for image verification and initial tagging.
- **Azure AI Translator:** Localizes English diagnoses into Hindi.
- **Azure AI Speech:** Converts Hindi text into clear audio guidance for accessibility.

## 🛠️ Technical Stack
- **Backend:** Flask (Python)
- **AI/ML:** TensorFlow Lite
- **Cloud:** Microsoft Azure (Vision, Speech, Translator)
- **Frontend:** Bootstrap 5, HTML5

## 📖 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Add your Azure Keys in `app.py`.
3. Run the server: `python app.py`

---
*Created by Prateek Gupta*
