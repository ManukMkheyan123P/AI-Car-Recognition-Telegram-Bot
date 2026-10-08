# 🚗 AI Car Recognition Telegram Bot

### Deep Learning | Computer Vision | Transfer Learning | Python

An experimental AI-powered Telegram bot that recognizes car models from photographs using **TensorFlow, Keras, and EfficientNetB0**.

> **Project Status: Prototype / Proof of Concept**
>
> This project is an educational machine learning prototype created to demonstrate the integration of computer vision, transfer learning, and a Telegram bot. It is not a production-ready vehicle recognition system. Predictions may be inaccurate, and further training, evaluation, and optimization are required.

## 📌 Project Overview

The goal of this project is to develop a Telegram bot capable of identifying car makes and models from user-submitted photographs.

The bot uses a convolutional neural network based on **EfficientNetB0**, a pretrained deep learning architecture, adapted to the Stanford Cars classification task.

Instead of training the entire network from scratch, the project applies **Transfer Learning**, using pretrained visual features to support vehicle recognition.

The current implementation supports **196 car model categories**, including distinctions between certain models, body styles, and production years.

## ⚙️ How It Works

1. **Image Upload:** The user sends a photograph of a car to the Telegram bot.
2. **Image Preprocessing:** The image is converted to RGB, resized to the model's expected input dimensions, and prepared as a NumPy array.
3. **Feature Extraction:** EfficientNetB0 processes the image and extracts visual features.
4. **Classification:** The trained classification head generates scores for 196 car categories.
5. **Prediction:** The model selects the category with the highest predicted probability.
6. **Telegram Response:** The bot returns the predicted vehicle model, confidence score, and top three predictions.

### Example Response

```text
🚘 Predicted Car: Ferrari 458 Italia Convertible 2012

📊 Model Confidence: 45.94%

🏆 Top-3 Predictions:

1. Ferrari 458 Italia Convertible 2012 — 45.94%
2. Ferrari California Convertible 2012 — 10.96%
3. Chevrolet Corvette ZR1 2012 — 10.20%
```

*This is an example from prototype testing. The confidence score is a model output, not a guarantee of correctness.*

## 🧠 Machine Learning Model

| Component | Description |
|---|---|
| Architecture | EfficientNetB0 |
| Framework | TensorFlow / Keras |
| Approach | Transfer Learning |
| Dataset | Stanford Cars |
| Classification Task | Fine-grained vehicle recognition |
| Number of Classes | 196 |
| Input | RGB vehicle images |
| Output | Predicted car category and confidence scores |

### Training Approach

The project uses EfficientNetB0 with pretrained weights and a classification head adapted to the vehicle dataset.

The pretrained network provides general visual features, while the classification head is trained to distinguish between car categories.

In the initial training experiment, the model achieved approximately **46% validation accuracy**.

This result demonstrates that the model can learn useful visual patterns, but its performance remains limited and requires further improvement.

## 🛠️ Technologies Used

- **Python** — application development
- **TensorFlow / Keras** — deep learning model
- **EfficientNetB0** — pretrained CNN architecture
- **NumPy** — image array processing
- **Pillow** — image preprocessing
- **python-telegram-bot** — Telegram Bot API integration
- **GitHub** — source code hosting and version control

## 📂 Project Structure

```text
Car-Recognition-Telegram-Bot/
│
├── bot.py
├── car_model.keras
├── class_names.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

- `bot.py` — Telegram bot logic and prediction pipeline
- `car_model.keras` — trained deep learning model
- `class_names.pkl` — mapping between class indices and car names
- `requirements.txt` — Python dependencies
- `README.md` — project documentation
- `.gitignore` — files excluded from version control

*If the trained model is hosted externally due to file size, download it and place it alongside `bot.py` before running the application.*

## 🚀 Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Car-Recognition-Telegram-Bot.git
cd Car-Recognition-Telegram-Bot
```

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure the Telegram Bot

Create a Telegram bot using [@BotFather](https://t.me/BotFather) and obtain your API token.

Set the token as an environment variable.

**Windows PowerShell:**

```powershell
$env:TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
```

Never commit a real Telegram token to GitHub.

### 4. Run the Application

```bash
python bot.py
```

### 5. Test the Bot

Open the bot in Telegram, send `/start`, and upload a photograph of a car.

The bot will attempt to identify the vehicle and return its top three predictions.

## ⚠️ Prototype Limitations

This project is currently an experimental prototype.

Known limitations include:

- **Limited classification accuracy:** The model may confuse visually similar car models.
- **Dataset limitations:** Performance depends on the diversity and quality of training images.
- **Confidence uncertainty:** Predicted probabilities should not be interpreted as guaranteed accuracy.
- **Limited coverage:** The model only recognizes categories represented in its training dataset.
- **Image sensitivity:** Camera angles, lighting, backgrounds, and image quality can affect predictions.
- **Local execution:** The current bot runs locally and requires the Python process to remain active.

The model has not been validated for production deployment.

## 🔮 Future Improvements

- Fine-tune additional EfficientNetB0 layers
- Expand and improve the training dataset
- Apply data augmentation and regularization
- Evaluate per-class precision, recall, and F1-score
- Improve preprocessing consistency
- Add confidence thresholds for uncertain predictions
- Deploy the Telegram bot to a cloud environment
- Add automated testing and logging

## 🎯 Learning Objectives

This project was developed to gain practical experience in:

- Convolutional Neural Networks (CNNs)
- Transfer Learning
- Image classification
- Model training and evaluation
- Saving and loading trained models
- Integrating machine learning with Telegram
- Building an end-to-end AI prototype

## 👨‍💻 Author

**Manuk Mkheyan**

Python & Machine Learning Developer

[GitHub Profile](https://github.com/ManukMkheyan123P)

---

**Disclaimer:** This project is intended for educational and experimental purposes. It demonstrates a working machine learning application, but its predictions should not be relied upon for professional vehicle identification.
