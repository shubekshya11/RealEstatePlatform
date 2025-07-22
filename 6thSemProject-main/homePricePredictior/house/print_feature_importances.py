import pickle
import sys
import os

# Add the directory containing your custom DecisionTreeRegressor to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_models'))

from decision_tree import DecisionTreeRegressor # Import your custom class


def print_importances():
    try:
        # Load feature names
        feature_names_path = 'house/ml_models/feature_names.pkl'
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)

        # Load the decision tree model
        model_path = 'house/ml_models/saved_models/decision_tree.pkl'
        # When loading custom class instances, specify the module
        with open(model_path, 'rb') as f:
            decision_tree_model = pickle.load(f)

        # Retrieve and sort feature importances
        importance = decision_tree_model.feature_importances_
        sorted_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

        print('\nFeature Importances (sorted):\n')
        for name, score in sorted_importance:
            print(f'{name}: {score:.4f}')

    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}. Make sure the file exists and the path is correct relative to the project root.")
    except ImportError as e:
        print(f"Error importing custom class: {e}. Make sure the path is correct and the class is defined.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Ensure the script is run from the correct directory (e.g., the manage.py directory)
    # This script assumes it's run from the directory containing the 'house' folder
    # If not, adjust the paths to the pickle files.
    print_importances() 