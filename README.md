# VeriNews 📰🔍  
**Fake News Detection using Machine Learning & NLP**

VeriNews is an end-to-end machine learning project that classifies news headlines/articles as **REAL** or **FAKE** using Natural Language Processing (NLP) techniques and a probabilistic classifier.

This project demonstrates the complete ML pipeline — from raw data preprocessing to model evaluation and visualization — showcasing how AI can be applied to detect misinformation.

---

## ✨ Features

- Text preprocessing (lowercasing, punctuation removal, stopword filtering, stemming)
- Feature extraction using **TF-IDF vectorization**
- Model training using **Multinomial Naive Bayes**
- Evaluation using:
  - Accuracy score
  - Confusion matrix
  - Classification report
- Visualization with **Matplotlib** and **Seaborn**
- Handles large real-world dataset (~100MB)

---

## 📂 Dataset

This project uses the **Fake and Real News Dataset** from Kaggle:

🔗 https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

The dataset includes:
- `Fake.csv`
- `True.csv`

These files are combined and labeled into a unified dataset:
- `text` → news content  
- `label` → `FAKE` or `REAL`

> ⚠️ Note: Due to the dataset size (~100MB), the dataset is not included in this repository.


## 📊 Model Performance

The model achieves strong performance on real-world news data, with an overall accuracy of approximately **95%**.

### 📌 Confusion Matrix Results

- True Positives (FAKE → FAKE): **4448**  
- True Positives (REAL → REAL): **4092**  
- False Positives (REAL predicted as FAKE): **155**  
- False Negatives (FAKE predicted as REAL): **285**

### 🔍 Key Insights

- High accuracy in distinguishing between REAL and FAKE news  
- Low misclassification rate relative to dataset size (~9000 samples)  
- Strong balance between precision and recall across both classes  
- Model generalizes well on real-world data  

![Confusion Matrix](confusion_matrix.png)

> 📌 *Confusion Matrix Visualization*  
> This heatmap illustrates the model’s prediction performance, showing correct classifications and errors across REAL and FAKE categories.


### 🛠️ Using Your Own Dataset

You can also use your own dataset as long as it follows this format:

```csv
text,label
"Some news headline or article",FAKE
"Another news example",REAL
