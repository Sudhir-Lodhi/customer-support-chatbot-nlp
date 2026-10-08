# 🤖 NLP-Based Customer Support Chatbot

An intelligent customer support chatbot built using **Natural Language Processing (NLP)** and **Machine Learning** to automatically understand customer queries, classify their intent, and provide appropriate predefined responses.

The system uses **TF-IDF vectorization** for text representation and a **Multinomial Naive Bayes classifier** for intent classification. A **Streamlit** web interface provides an interactive chatbot experience.

---

## 📌 Project Overview

Customer support systems receive a large number of repetitive queries related to orders, payments, refunds, delivery, accounts, invoices, and other common issues.

Handling these queries manually can be time-consuming. This project addresses the problem by developing an NLP-based chatbot that can automatically identify the intent behind a customer's message and return a relevant response.

### Example

**User:**

> Where is my order?

**Detected Intent:**

`track_order`

**Chatbot:**

> You can track your order using your order details or tracking information.

---

## 🎯 Objectives

The main objectives of this project are:

- To develop an NLP-based customer support chatbot.
- To preprocess and normalize natural-language customer queries.
- To convert text into numerical features using TF-IDF.
- To classify customer queries into predefined support intents.
- To use a machine learning model for automatic intent classification.
- To generate appropriate responses based on the predicted intent.
- To provide an interactive web interface using Streamlit.
- To evaluate the performance of the trained classification model.

---

## ✨ Key Features

- 💬 Interactive customer support chatbot
- 🧠 NLP-based intent classification
- 🔤 Text preprocessing
- 📊 TF-IDF feature extraction
- 🤖 Multinomial Naive Bayes classification
- 🎯 27 customer-support intents
- 📈 Model confidence display
- 🖥️ Streamlit web interface
- 💾 Pre-trained model saved using Joblib
- 🗂️ Separate training, testing, and validation datasets
- 🧹 Clear chat functionality
- 💡 Suggested customer queries

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │     User Query       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Text Preprocessing   │
                    │                      │
                    │ • Lowercasing        │
                    │ • Remove symbols     │
                    │ • Remove extra space │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   TF-IDF Vectorizer  │
                    │                      │
                    │ Text → Numerical     │
                    │ Features             │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Multinomial Naive    │
                    │ Bayes Classifier     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Intent Prediction   │
                    │                      │
                    │ Example:             │
                    │ track_order          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Response Generation  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Chat UI    │
                    └──────────────────────┘
```

---

## 🔄 NLP Pipeline

The chatbot follows the following processing pipeline:

### 1. User Input

The user enters a natural-language customer support query.

Example:

```text
I forgot my password
```

### 2. Text Preprocessing

The input text is normalized by:

- Converting text to lowercase
- Removing unnecessary special characters
- Removing extra whitespace

Example:

```text
I DON'T know how to cancel my order!!!
```

becomes:

```text
i dont know how to cancel my order
```

### 3. TF-IDF Vectorization

The cleaned text is converted into numerical feature vectors using **Term Frequency-Inverse Document Frequency (TF-IDF)**.

TF-IDF helps represent words based on their importance within the dataset.

### 4. Intent Classification

The TF-IDF vector is passed to a **Multinomial Naive Bayes** classifier.

The model predicts one of the predefined customer-support intents.

### 5. Response Generation

The predicted intent is mapped to a predefined chatbot response.

---

## 🧠 Machine Learning Model

### Multinomial Naive Bayes

The project uses **Multinomial Naive Bayes**, a probabilistic machine-learning algorithm commonly used for text classification.

It is suitable for this project because:

- It works well with text-based features.
- It performs efficiently on high-dimensional TF-IDF vectors.
- It is computationally lightweight.
- It is simple and suitable for intent classification.

---

## 📊 Dataset

The project uses the **Bitext Customer Service Dataset**, containing customer-service utterances classified into different intents.

### Dataset Split

| Dataset | Records |
|---|---:|
| Training | 6,539 |
| Testing | 818 |
| Validation | 818 |
| **Total** | **8,175** |

The dataset contains **27 customer-support intents**.

### Intent Categories

```text
cancel_order
change_order
change_shipping_address
check_cancellation_fee
check_invoice
check_payment_methods
check_refund_policy
complaint
contact_customer_service
contact_human_agent
create_account
delete_account
delivery_options
delivery_period
edit_account
get_invoice
get_refund
newsletter_subscription
payment_issue
place_order
recover_password
registration_problems
review
set_up_shipping_address
switch_account
track_order
track_refund
```

---

## 📈 Model Performance

The trained Multinomial Naive Bayes classifier was evaluated on the separate testing dataset.

### Test Accuracy

**99.14%**

```text
Accuracy: 0.9914
```

The classification report also showed approximately **0.99 macro-average precision, recall, and F1-score**, indicating consistently strong performance across the 27 intent classes.

> Note: The reported 99.14% accuracy is the performance on the held-out testing dataset and should not be interpreted as perfect performance on all possible real-world customer queries.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset handling |
| Scikit-learn | Machine learning and TF-IDF |
| Multinomial Naive Bayes | Intent classification |
| Joblib | Saving and loading trained models |
| Streamlit | Web-based chatbot interface |
| Git | Version control |
| GitHub | Source-code hosting |

---

## 📁 Project Structure

```text
customer-support-chatbot-nlp/
│
├── app/
│   └── app.py
│
├── dataset/
│   └── banking77/
│       ├── Bitext_Sample_Customer_Service_Training_Dataset.csv
│       ├── Bitext_Sample_Customer_Service_Testing_Dataset.csv
│       └── Bitext_Sample_Customer_Service_Validation_Dataset.csv
│
├── models/
│   ├── intent_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── chatbot.py
│   ├── evaluate_model.py
│   ├── preprocess.py
│   └── train_model.py
│
├── .gitignore
├── dataset_analysis.py
├── test_preprocessing.py
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sudhir-Lodhi/customer-support-chatbot-nlp.git
```

### 2. Navigate to the project directory

```bash
cd customer-support-chatbot-nlp
```

### 3. Install required Python packages

```bash
pip install pandas scikit-learn joblib streamlit matplotlib
```

---

## ▶️ Running the Chatbot

Start the Streamlit application using:

```bash
python -m streamlit run app/app.py
```

The application will open in your browser.

The default local address is:

```text
http://localhost:8501
```

---

## 🧪 Running the Terminal Chatbot

The project also includes a terminal-based chatbot.

Run:

```bash
python src/chatbot.py
```

Example:

```text
======================================
     CUSTOMER SUPPORT CHATBOT
======================================

You: Where is my order?

Detected Intent: track_order

Bot: You can track your order using your order details or tracking information.
```

---

## 🔬 Training the Model

To retrain the model using the training dataset:

```bash
python src/train_model.py
```

The trained files are saved in:

```text
models/
├── intent_model.pkl
└── tfidf_vectorizer.pkl
```

---

## 📊 Evaluating the Model

To evaluate the trained model using the testing dataset:

```bash
python src/evaluate_model.py
```

The evaluation script calculates:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report

---

## 💬 Example Queries

The chatbot can handle queries such as:

| User Query | Predicted Intent |
|---|---|
| Where is my order? | `track_order` |
| I forgot my password | `recover_password` |
| I want to cancel my order | `cancel_order` |
| How can I get a refund? | `get_refund` |
| What payment methods do you accept? | `check_payment_methods` |
| My payment is not working | `payment_issue` |
| I want to create an account | `create_account` |
| How long will delivery take? | `delivery_period` |

---

## 🚧 Limitations

Although the chatbot performs well on the test dataset, it has some limitations:

- It supports only the intents represented in the training dataset.
- Responses are predefined rather than dynamically generated
