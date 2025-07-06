# Email Spam Detection Project

## Overview

This project focuses on building a machine learning model to classify emails as spam or ham (non-spam). Using a dataset of labeled emails, the project implements data preprocessing, feature engineering, model training, and evaluation to create an effective spam detection system. The primary model used is a Multinomial Naive Bayes classifier, with performance metrics like accuracy, precision, recall, and F1-score evaluated to ensure robust performance.

![Email Spam Detection](https://miro.medium.com/v2/resize:fit:1400/0*j1wMZQ2je5P5DHvN)

## Project Objectives

- **Data Preparation**: Clean and preprocess email data for machine learning.
- **Feature Engineering**: Extract relevant features from email text using techniques like CountVectorizer.
- **Model Development**: Train and evaluate a spam detection model using the Multinomial Naive Bayes algorithm.
- **Performance Evaluation**: Assess model performance using metrics such as accuracy, precision, recall, and F1-score.
- **Model Optimization**: Fine-tune the model to improve predictive accuracy.
- **Deployment Considerations**: Explore strategies for deploying the model in real-world email filtering systems.

## Dataset

The dataset used is the [SMS Spam Collection Dataset](https://miro.medium.com/v2/resize:fit:1400/1*fL91mSzsFKTvJNTEZ_cHfQ.png), containing labeled SMS messages as spam or ham. It is loaded from a GitHub repository in the notebook (`spam.csv`). The dataset includes:

- **v1**: Label (ham or spam)
- **v2**: Email/message text
- Additional unnamed columns (handled during preprocessing)

## Prerequisites

To run the notebook, you need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `wordcloud`

Install the dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn wordcloud
```

## Project Structure

The Jupyter Notebook (`Email_Spam_Detection.ipynb`) is organized as follows:

1. **Introduction**: Overview of the spam detection problem and project goals.
2. **Data Exploration**: Initial analysis of the dataset, including shape and basic statistics.
3. **Data Preprocessing**: Cleaning the dataset, handling missing values, and transforming text data.
4. **Feature Engineering**: Converting text into numerical features using CountVectorizer.
5. **Model Training**: Implementing a Multinomial Naive Bayes classifier using a scikit-learn Pipeline.
6. **Model Evaluation**: Visualizing performance metrics (accuracy, precision, recall, F1-score) and confusion matrices.
7. **Spam Detection System**: A function to classify new emails as spam or ham.
8. **Conclusion**: Summary of findings and potential real-world applications.

![Machine Learning Pipeline](https://miro.medium.com/v2/resize:fit:1400/1*WA9aceQugVlBS81r2a7Snw.png)

## How to Run

1. Clone the repository or download the `Email_Spam_Detection.ipynb` file.
2. Ensure all dependencies are installed (see Prerequisites).
3. Open the notebook in Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook Email_Spam_Detection.ipynb
   ```
4. Run the cells sequentially to load the dataset, preprocess data, train the model, and evaluate performance.
5. Use the `detect_spam` function to test the model on new email text.

## Results

The Multinomial Naive Bayes model achieved:

- **Train Accuracy**: 99.35%
- **Test Accuracy**: 98.49%
- **Test Recall**: 98.49% (chosen as the primary metric for minimizing false negatives)

The model effectively distinguishes between spam and ham emails, with high recall ensuring minimal legitimate emails are misclassified as spam.

![Confusion Matrix Example](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRHumc9hD3GoGT-pqVudZxOjxaxGpJN9izz4w&s)

## Usage Example

To classify a new email:

```python
sample_email = "We are pleased to inform you that your email address has been selected as a winner of $10,000 in our annual Mega Prize Draw!"
result = detect_spam(sample_email)
print(result)  # Output: This is a Spam Email!
```

## Future Improvements

- Explore advanced models like Support Vector Machines or Deep Learning (e.g., LSTM).
- Incorporate additional features such as email metadata (e.g., sender, subject line).
- Deploy the model as an API for real-time email filtering using frameworks like Flask or FastAPI.
- Enhance preprocessing with advanced NLP techniques (e.g., TF-IDF, word embeddings).

## Conclusion

This project demonstrates a robust approach to email spam detection using machine learning. By leveraging the Multinomial Naive Bayes algorithm and thorough evaluation, the system achieves high accuracy and recall, making it suitable for real-world email filtering applications.

![_](https://cdn.prod.website-files.com/659fa592476e081fbd5a3335/669f84768899d32f200f7556_spamz.png)
