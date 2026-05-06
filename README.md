# 🎬 Movie Recommender System

A modular machine learning recommender system built using the MovieLens dataset.  
It combines **content-based filtering** and **item-item collaborative filtering** to generate personalized movie recommendations.

---

## 🚀 Features

- Content-based filtering using TF-IDF on movie genres and user tags  
- Item-item collaborative filtering using cosine similarity  
- Rating prediction for unseen movies  
- Evaluation using Mean Absolute Error (MAE)  
- Modular and reusable Python codebase  

---

## 📊 Dataset

MovieLens Small Dataset containing:

- `movies.csv` → movie metadata (title, genres)
- `ratings.csv` → user-movie ratings
- `tags.csv` → user-generated tags for movies

---

## 🧠 Methods

### 1. Content-Based Filtering
- Movie profiles are created by combining genres and user tags  
- TF-IDF is used to convert text into feature vectors  
- Cosine similarity is used to find similar movies  

### 2. Collaborative Filtering (Item-Item)
- Builds a user–movie rating matrix  
- Computes similarity between movies using cosine similarity  
- Predicts ratings based on weighted similarities from user history  

---

## 📈 Evaluation

The system is evaluated using:

- **Mean Absolute Error (MAE)** for rating prediction accuracy  
- Example MAE: ~0.7–0.9 (depending on train/test split)

---

## 🧪 Example Output

### Content-Based Recommendation
Input: *Toy Story (1995)*

Output:
- A Bug’s Life  
- Antz  
- The Emperor’s New Groove  
- ...

### Rating Prediction Example

Actual: 5.0 → Predicted: 4.40
Actual: 4.0 → Predicted: 4.36

---

## ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

---

## Project Structure

src/        → Core implementation (models + logic)
data/       → MovieLens dataset
main.py     → Entry point for full pipeline
notebooks/  → Exploratory analysis and visualization