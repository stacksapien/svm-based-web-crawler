# URL Classification Using Support Vector Machine (SVM)

This project implements a Support Vector Machine (SVM)-based focused crawler to classify blog URLs into specific categories such as Technology, Travel, and Lifestyle. The implementation follows the methodology outlined in the paper:

Baweja, V.R., Bhatia, R., Kumar, M. (2020). Support Vector Machine-Based Focused Crawler. In: Ranganathan, G., Chen, J., Rocha, Á. (eds) _Inventive Communication and Computational Technologies_. Lecture Notes in Networks and Systems, vol 89. Springer, Singapore. [https://doi.org/10.1007/978-981-15-0146-3_63](https://doi.org/10.1007/978-981-15-0146-3_63)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data Collection](#data-collection)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Classification](#classification)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Overview

The objective of this project is to classify blog URLs into predefined categories using an SVM classifier. By analyzing the structure and content of URLs, the model predicts the category to which a given URL belongs. This approach is beneficial for focused crawling, content categorization, and information retrieval tasks.

## Installation

To set up the project environment, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone git@github.com:stacksapien/svm-based-web-crawler.git
   cd svm-based-web-crawler
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv env
   ```

3. **Activate the virtual environment**:

   - On Windows:

     ```bash
     .\\env\\Scripts\\activate
     ```

   - On macOS/Linux:

     ```bash
     source env/bin/activate
     ```

4. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After setting up the environment, you can run the URL classification script:

```bash
python main.py
```

This script will preprocess the URLs, extract features, train the SVM model, and classify new URLs into their respective categories.

## Data Collection

The dataset comprises sample blog URLs categorized into Technology, Travel, and Lifestyle. Each URL is labeled accordingly to serve as training data for the SVM classifier.

## Preprocessing

Preprocessing involves cleaning and tokenizing URLs:

- Removing protocols (`http`, `https`) and `www`.
- Replacing non-alphanumeric characters with spaces.
- Converting text to lowercase.

This standardization facilitates effective feature extraction.

## Feature Extraction

The preprocessed URLs are transformed into numerical feature vectors using Term Frequency-Inverse Document Frequency (TF-IDF) vectorization. This technique quantifies the importance of terms within the URLs, enabling the SVM classifier to discern patterns associated with each category.

## Model Training

The dataset is divided into training and testing sets. An SVM classifier with a linear kernel is trained on the training data to learn the distinctions between categories.

## Classification

The trained SVM model predicts the category of new URLs based on their structural features. The `classify_url` function processes a new URL and outputs its predicted category.

## Evaluation

The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. These metrics provide insights into the classifier's effectiveness in categorizing URLs.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
