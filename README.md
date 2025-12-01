**#Sentiment Analysis on Movie Reviews Using NLP**

📝** Project Overview**
Sentiment Analysis is a Natural Language Processing (NLP) technique used to determine the emotional tone behind text data.
In this project, we analyze movie reviews and classify them as Positive or Negative using Logistic Regression along with TF-IDF vectorization.
Additionally, we visualize frequent words using WordClouds.

🎯**Objectives**

Preprocess and clean text data

Convert text into numerical features using TF-IDF

Build a sentiment classification model using Logistic Regression

Perform EDA and visualize frequent words using WordClouds

Predict sentiment on new/custom reviews

📂** Dataset**

The dataset contains movie reviews with sentiment labels.

Review	Sentiment
"Amazing movie with brilliant acting"	Positive
"The film was boring and slow"	Negative
🔧 Data Preprocessing

**Applied text preprocessing techniques:**

Remove non-alphabetical characters
Convert text to lowercase
Remove stopwords
Apply stemming using PorterStemmer

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

📊 Exploratory Data Analysis (EDA)
WordCloud Visualization:

Positive Reviews → good, excellent, amazing
Negative Reviews → boring, slow, waste

plt.figure(figsize=(10,5))
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud - Positive Reviews")
plt.show()



🔠 **Feature Engineering**
Used TF-IDF Vectorization to convert text into numeric representation.

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned_review']).toarray()
y = df['sentiment']

🤖** Model Training**

Model: Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

📈 **Model Evaluation**
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

Results
Accuracy: ~ 74%
Correct predictions: 74 out of 100

🧪 Testing Custom Review
sample = ["The film was boring and slow, I hated it"]
sample_clean = [clean_text(sample[0])]
sample_vector = tfidf.transform(sample_clean).toarray()
print("Predicted Sentiment:", model.predict(sample_vector)[0])
Output: Negative

🏁 **Conclusion:**
This project demonstrates the complete workflow of sentiment analysis including:
Text preprocessing & cleaning
Vocabulary visualization using WordClouds
TF-IDF feature extraction

Sentiment prediction with Logistic Regression

Key Takeaways

NLP performance depends on dataset size & quality

74% accuracy is reasonable for small datasets

WordClouds provide insights into commonly used words

🚀 **Future Improvements**
Use advanced models like Naive Bayes, SVM, LSTM, BERT.
Deploy using Streamlit / Flask.

💡 **Technologies Used**

Python
Pandas, NumPy, Matplotlib, WordCloud
Scikit-learn
NLP (TF-IDF, Stemming, Stopwords)





