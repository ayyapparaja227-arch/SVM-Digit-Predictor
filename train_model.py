import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_and_save_model():
    # Load dataset
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model (SVM with Polynomial kernel as it had best accuracy)
    svm_poly = SVC(kernel='poly', C=1.0, gamma='scale')
    svm_poly.fit(X_train, y_train)

    # Evaluate model
    y_pred = svm_poly.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained. Accuracy: {accuracy:.4f}")

    # Save model
    joblib.dump(svm_poly, 'model.pkl')
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train_and_save_model()
