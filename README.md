# Verinews-fake-news-classifier
VeriNews is a machine learning project that can tell if a news headline is real or fake. It uses Python and some basic natural language processing to understand the text and make predictions. You can even try it out with your own headline using a simple web app.



## 🔍 Features

- Preprocesses text (lowercase, remove stopwords/punctuation, stemming)
- Converts text to features using TF-IDF
- Trains a Multinomial Naive Bayes classifier
- Evaluates with confusion matrix, classification report, and ROC curve
- Exports model using `joblib`
- Optional: Simple UI with Streamlit for user input

---

## 📂 Dataset

Use the [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) or any CSV with at least:
- `text`: the article/headline
- `label`: either "FAKE" or "REAL"

---

## 🚀 How to Run

1. Clone this repo:
git clone https://github.com/msahal-13/verinews-fake-news-classifier.git
cd verinews-fake-news-classifier

2. Install dependencies:
pip install -r requirements.txt

3. Add your dataset as `fake_or_real_news.csv`

4. Run the classifier:
python verinews_classifier.py
