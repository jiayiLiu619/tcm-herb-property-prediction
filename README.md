# TCM Herb Property Prediction with Machine Learning

This project explores machine learning approaches for predicting Traditional Chinese Medicine (TCM) herb properties from aggregated chemical component features.

Using data derived from ETCM, the project models herb attributes such as temperature, flavor, and meridian tropism as binary classification tasks. Multiple machine learning approaches were explored, including Neural Networks, Logistic Regression, and Gradient Boosting, with Neural Networks achieving the strongest overall performance.

## Highlights
- Predicted 24 herb-property targets from chemical component features
- Applied mean pooling to aggregate variable-length compound information into herb-level representations
- Built a neural network for binary classification across multiple targets
- Used cross-validation for hyperparameter tuning
- Implemented early stopping to reduce overfitting
- Compared Neural Network, Logistic Regression, and Gradient Boosting baselines

## Method
- Dataset built from ETCM herb and compound data
- 21 chemical features used as model inputs
- Herb properties grouped into temperature, flavor, and meridian tropism categories
- Mean pooling used to convert variable-length compound sets into fixed-length herb representations
- Neural network architecture: 21 -> 30 -> 60 -> 1
- ReLU hidden activations, sigmoid output, BCELoss, Adam optimizer

## Results
The project found that Neural Networks achieved the best overall performance in terms of Hamming Loss among the tested methods.

Performance varied by target property: some targets such as sour and hot reached around 90% accuracy, while others such as sweet and liver were more difficult and remained closer to 60–70%.

## My Contribution
This was a team machine learning project. My contribution focused on implementing the neural network pipeline, training workflow, and evaluation logic in PyTorch.

## Files
- `main.py`: main training and evaluation workflow
- `net.py`: neural network definition and training logic
- `TCM_Herb_Property_Prediction_Report.pdf`: project report

## Run
1. Install required dependencies
2. Prepare the dataset in the expected format
3. Run the project with:
   ```bash
   python main.py
