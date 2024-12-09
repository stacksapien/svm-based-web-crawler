import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Sample blog URLs
urls = [
    "https://www.servicenow.com/blogs/2022/3-leaders-manufacturing-innovation",
    "https://www.servicenow.com/blogs/2022/how-detection-engineering-keeps-us-safe",
    "https://www.servicenow.com/blogs/2022/servicenow-research-neurips-2022",
    "https://theworldtravelguy.com/mount-karangetang-volcano-hike/",
    "https://thediscoverynut.com/turkey-vs-morocco-travel/",
    "https://www.eggcanvas.com/blog/2021/skincare-routine",
    "https://thedaileigh.com/what-to-wear/2024/11/20/winter-leisure-capsule-wardrobe-2024-2025"
]

# Corresponding labels: 0 for technology, 1 for travel, 2 for lifestyle
labels = [0, 0, 0, 1, 1, 2, 2]

# Preprocessing function to clean and tokenize URLs
def preprocess_url(url):
    # Remove protocol (http, https) and 'www'
    url = re.sub(r'https?://(www\.)?', '', url)
    # Replace non-alphanumeric characters with spaces
    url = re.sub(r'[^a-zA-Z0-9]', ' ', url)
    # Convert to lowercase
    url = url.lower()
    return url

# Preprocess all URLs
preprocessed_urls = [preprocess_url(url) for url in urls]

# Feature extraction using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_urls)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to classify new URLs
def classify_url(url):
    preprocessed = preprocess_url(url)
    features = vectorizer.transform([preprocessed])
    prediction = svm_classifier.predict(features)
    categories = {0: 'Technology', 1: 'Travel', 2: 'Lifestyle'}
    return categories[prediction[0]]

# Classify new URLs
new_urls = [
    "https://www.vox.com/future-perfect/361749/universal-basic-income-sam-altman-open-ai-study",
    "https://www.nomadicmatt.com/travel-guides/japan-travel-tips/",
    "https://cupofjo.com/2020/06/25/monastery-skincare-giveaway/"
]

for url in new_urls:
    print(f"URL: {url} -> Category: {classify_url(url)}")
