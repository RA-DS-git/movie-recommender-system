# 🎬 Movie Recommender System

A modular machine learning recommender system built using the MovieLens dataset.

It combines **content-based filtering** and **item-item collaborative filtering** to model user preferences and generate personalized movie recommendations.

---

## 🚀 Features

- Content-based filtering using TF-IDF on movie genres and user tags  
- Item-item collaborative filtering using cosine similarity on user ratings  
- Rating prediction for unseen movies  
- Evaluation using Mean Absolute Error (MAE)  
- Modular and extensible Python codebase  

---

## 📊 Dataset

MovieLens Small Dataset:

- `movies.csv` → movie metadata (title, genres)
- `ratings.csv` → user-movie ratings
- `tags.csv` → user-generated tags

---

## 🧠 Methods

### 1. Content-Based Filtering
- Builds movie profiles using genres + tags  
- TF-IDF converts text into feature vectors  
- Cosine similarity retrieves similar movies  

### 2. Collaborative Filtering (Item-Item)
- Constructs a user–movie rating matrix  
- Computes item-item similarity using cosine similarity  
- Predicts ratings using similarity-weighted averages  

---

## 📈 Key Findings

- User ratings are skewed toward higher values (3–5 range)  
- Popular movies dominate interaction data (popularity bias)  
- Content-based filtering captures genre-level similarity but misses deeper semantics  
- Collaborative filtering performs reasonably well but suffers from sparsity and cold-start issues  
- Model achieves MAE ≈ 0.73 on held-out users  

---

## 🧪 Example Outputs

### 🎯 Content-Based Recommendation (Toy Story)

- A Bug’s Life  
- Antz  
- Emperor’s New Groove  
- Rocky and Bullwinkle  
- Guardians of the Galaxy Vol. 2  

---

### 📊 Rating Prediction Examples

```text
Actual: 5.0 → Predicted: 4.40
Actual: 4.0 → Predicted: 4.36
Actual: 5.0 → Predicted: 4.38
```

---

## 📉 Evaluation

The model is evaluated using:

- **MAE (Mean Absolute Error)** for rating prediction accuracy  
- Example result: ~0.73

---

## ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py
```

---

## Project Structure

```text
src/            Core implementation (models + logic)
data/           MovieLens dataset
notebooks/      EDA + model exploration
main.py         Pipeline entry point
```