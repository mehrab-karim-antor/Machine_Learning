# ------------------------------------------------------------
# KNN CLASSIFIER (FROM SCRATCH) - IMDb Style Movie Genre Prediction
# ------------------------------------------------------------

import numpy as np
from collections import Counter

# ------------------------------------------------------------
# 1. Create a Fake IMDB Dataset
# ------------------------------------------------------------
# Features:
#   Rating (1–10)
#   Action Scenes Count
#   Comedy Scenes Count
#   Drama Scenes Count
#
# Target (Genre):
#   0 = Action
#   1 = Comedy
#   2 = Drama

X = np.array([
    [8.5, 40, 5, 3],    # Action
    [7.0, 30, 4, 4],    # Action
    [6.5, 25, 3, 6],    # Action
    [5.0, 10, 30, 5],   # Comedy
    [6.0, 8, 35, 4],    # Comedy
    [7.5, 5, 25, 6],    # Comedy
    [8.7, 5, 3, 40],    # Drama
    [7.8, 6, 4, 30],    # Drama
    [6.2, 4, 5, 35]    # Drama
    
])

y = np.array([
    0, 0, 0,   # Action
    1, 1, 1,   # Comedy
    2, 2, 2    # Drama
])

# ------------------------------------------------------------
# 2. KNN Class Implementation (K = 5)
# ------------------------------------------------------------

class KNNClassifier:

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict_one(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        predictions = [self.predict_one(x) for x in X]
        return np.array(predictions)


# ------------------------------------------------------------
# 3. Train the Model
# ------------------------------------------------------------
knn = KNNClassifier(k=5)
knn.fit(X, y)

# ------------------------------------------------------------
# 4. Predict Movie Genre for a New Movie
# ------------------------------------------------------------
# Example new movie:
# Rating = 7.8
# Action scenes = 12
# Comedy scenes = 25
# Drama scenes = 8

new_movie = np.array([[7.8, 112, 5525, 12348]])
prediction = knn.predict(new_movie)

# ------------------------------------------------------------
# 5. Show Output
# ------------------------------------------------------------
genre_map = {0: "Action", 1: "Comedy", 2: "Drama"}

print("New Movie Features:", new_movie)
print("Predicted Genre:", genre_map[prediction[0]])
