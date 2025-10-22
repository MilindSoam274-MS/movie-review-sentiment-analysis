# Movie Review Sentiment Analysis with Deep Learning

## Project Overview

This project is a comprehensive exploration of deep learning techniques for sentiment analysis. The goal is to classify movie reviews from the IMDB dataset as either positive or negative. Several neural network architectures are implemented and compared to evaluate their effectiveness. The project also explores the impact of using different pre-trained word embeddings, specifically GloVe and Word2Vec.

---

## Publication

The methodology and results of this project were formally published in IEEE Xplore. This publication provides an in-depth analysis of the models and their performance.

* **Title:** [Sentiment Analysis Using Deep Learning: A Comparative Study]
* **Conference/Journal:** [2022 Second International Conference on Computer Science, Engineering and Applications (ICCSEA)]
* **Link:** [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9936323&isnumber=9936049]

**Citation:**
[D. Soam and S. Thakur, "Sentiment Analysis Using Deep Learning: A Comparative Study," 2022 Second International Conference on Computer Science, Engineering and Applications (ICCSEA), Gunupur, India, 2022, pp. 1-6, doi: 10.1109/ICCSEA54677.2022.9936323. keywords: {Deep learning;Computer science;Sentiment analysis;Companies;Tag clouds;Motion pictures;Data models;Deep Learning;Recurrent Neural Network;LSTM;BiLSTM;Word Embedding;Glove},
]

---

## Dataset & Word Embeddings

* **Dataset:** The project uses the well-known **IMDB Movie Review Dataset**, which contains 50,000 movie reviews evenly split into training and testing sets.
* **Word Embeddings:**
    * **GloVe (Global Vectors for Word Representation):** Utilizes pre-trained word vectors to capture semantic relationships.
    * **Word2Vec:** Another popular technique for learning word embeddings from text.

---

## Models Explored

This repository contains multiple experiments comparing different neural network architectures and word embeddings.

* **Neural Network Architectures:**
    * **LSTM (Long Short-Term Memory)**
    * **BiLSTM (Bidirectional LSTM)**
    * **CNN (Convolutional Neural Network)**

---

## Results & Best Performing Model

After extensive experimentation, the **Bidirectional LSTM (BiLSTM)** model demonstrated the best performance for this sentiment analysis task.

The final evaluation and testing of this top-performing model can be found in the **`Testing_BiLSTM.ipynb`** notebook.

## Notebooks Overview

This repository includes several Jupyter Notebooks, each representing a different experiment in this comparative study:
* `Glove_BiLSTM.ipynb`
* `Glove_final.ipynb`
* `Improved_Glove(BiLSTM).ipynb`
* `LSTM(64)_Glove_Graph.ipynb`
* `New_improved_BiLSTM(Glove).ipynb`
* `Word2Vec_BiLSTM.ipynb`
* `Word2Vec_CNN.ipynb`
* `Word2Vec_LSTM.ipynb`
* **`Testing_BiLSTM.ipynb` (Contains the final testing of the best model)**

---

## Tools and Libraries

* **Python**
* **TensorFlow / Keras**
* **NumPy & Pandas**
* **Scikit-learn**
* **Jupyter Notebook**