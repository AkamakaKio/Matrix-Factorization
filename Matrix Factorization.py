from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load dataset
data = Dataset.load_builtin('ml-100k')

# Use SVD algorithm
model = SVD()

# Perform cross validation
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
