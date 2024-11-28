### Team Members

1. Satya Mani Srujan Dommeti #A20594429
2. Arjun Singh  #A20577962
3. Akshitha Reddy Kuchipatla #A20583607 
4. Vamsi Krishna  #A20543669
   
This repository contains a Python implementation of the ElasticNet linear regression model from scratch. The implementation includes features for training, evaluation, and visualization of results, without using prebuilt ElasticNet libraries like scikit-learn.

The code is designed to be modular and reusable, making it suitable for academic purposes or as a foundational implementation for learning about ElasticNet regularization.


# Model Implementation

The implementation of the ElasticNet regression model is designed from first principles to provide a comprehensive understanding of the optimization process, making it suitable for academic and practical learning purposes. The model incorporates both L1 (Lasso) and L2 (Ridge) regularization techniques, controlled by the `alpha` and `l1_ratio` hyperparameters, to balance feature selection and shrinkage. 

The model is implemented in Python using foundational libraries like NumPy for numerical operations, and Matplotlib for visualizations. A custom gradient descent algorithm is employed to iteratively optimize the coefficients and intercept, ensuring convergence through a combination of learning rate and tolerance criteria. 

The implementation includes key functionalities such as soft-thresholding for L1 penalty, computation of the cost function (combining Mean Squared Error and regularization penalties), and metrics like MSE and R² for evaluating model performance. Additionally, it offers utility methods for plotting cost convergence, feature importance, and predicted vs. actual values, making it a well-rounded and interpretable solution.

### What Does the Model Do and When Should It Be Used?
The ElasticNet regression model performs linear regression while balancing feature selection (via L1 regularization) and coefficient shrinkage (via L2 regularization). This dual regularization makes it particularly effective for datasets where multicollinearity exists among features or when a subset of features is expected to have negligible influence on the target variable.

Use cases include:

Predictive modeling with high-dimensional data.
Feature selection when the number of predictors is greater than the number of observations.
Scenarios where neither Lasso nor Ridge alone performs well.

### How Was the Model Tested?
The model was tested using the California Housing Dataset, a publicly available benchmark dataset. The following validation techniques ensured correctness:

Synthetic Data Testing: Known datasets with defined coefficients were used to verify if the model correctly estimated the relationships.
Comparison with scikit-learn: Outputs (predicted values, coefficients, cost convergence) were compared with those from the ElasticNet model in scikit-learn.
Metrics Evaluation:
Training and validation errors (Mean Squared Error).
R² scores to measure goodness of fit.
Cross-Validation: Used to ensure robustness and reduce overfitting.

User-Exposed Parameters
Which Parameters Can Be Tuned?
The following hyperparameters are available for tuning:

alpha: Regularization strength. Higher values increase regularization, potentially improving generalization but risking underfitting.
l1_ratio: The balance between L1 and L2 penalties. Ranges from 0 (pure Ridge) to 1 (pure Lasso). Default: 0.6.
learning_rate: Controls the step size for gradient descent. Lower values ensure stability; higher values speed up convergence but risk overshooting.
max_iterations: Number of iterations for gradient descent. Default: 2000.
tolerance: Determines the stopping condition based on gradient magnitude.

### Known Limitations
Does the Model Struggle With Certain Inputs?
Highly Correlated Features: ElasticNet generally handles multicollinearity well, but extreme correlations can still cause instability in coefficient estimates.
Non-Linear Relationships: The model assumes a linear relationship between predictors and the target. Non-linearity in the data can lead to poor predictions.
Sparse Datasets: While ElasticNet is designed for sparse datasets, extremely high sparsity may require additional preprocessing, such as imputation or dimensionality reduction.

## Step-by-Step Guide to Use the ElasticNet Implementation:
### Set Up Your Environment
Before running the code, ensure you have Python installed (preferably version 3.8 or higher) and the following dependencies installed:

NumPy
Pandas 
SciPy

### Prepare Your Dataset

Make sure your dataset is in a suitable format:
Features (X): A 2D array or DataFrame where rows are samples, and columns are features.
Target (y): A 1D array or Series representing the target variable.
###Initialize the ElasticNet Model
Create an instance of the ElasticNet class, specifying parameters like alpha, l1_ratio, learning_rate, and max_iterations:
model = ElasticNet(alpha=0.01, l1_ratio=0.6, learning_rate=0.1, max_iterations=2000)
### Tune Parameters
Experiment with different values for the following parameters to optimize performance:

alpha: Regularization strength.
l1_ratio: The balance between L1 and L2 penalties.
learning_rate: Controls the speed of convergence.
max_iterations: Number of iterations for gradient descent.
### Use Custom Data
If you have your own dataset, replace X and y with your features and target variable. Ensure proper preprocessing like handling missing values or scaling features if needed.
Make sure to change the name from "small_test.csv" to your local data set name.
### Results:
<img width="1468" alt="Screenshot 2024-11-28 at 1 10 27 AM" src="https://github.com/user-attachments/assets/0929aec6-33e1-4001-9e74-9cd6b0a7b3ac">
### Challenges Faced
Implementing ElasticNet from Scratch:
Designing a custom gradient descent algorithm to balance L1 and L2 regularization while ensuring convergence required significant effort. Achieving accurate soft-thresholding for the L1 penalty and integrating it with the L2 penalty was challenging.

Parameter Tuning:
Selecting optimal values for hyperparameters like alpha, l1_ratio, learning_rate, and tolerance required extensive experimentation. Without automated tools like GridSearchCV, tuning parameters manually was time-intensive.

Dataset Compatibility:
Ensuring the implementation could handle diverse datasets, including those with missing values, non-numeric features, or extreme multicollinearity, involved additional preprocessing steps.

Validation and Debugging:
Testing the model on synthetic and real-world datasets (e.g., California Housing) and debugging discrepancies with scikit-learn's ElasticNet outputs required careful comparisons and fine-tuning.

Performance Optimization:
Iterating through a large number of data points and features during gradient descent caused computational overhead. Optimizing the implementation for efficiency while maintaining accuracy was a critical challenge.

Future Work
Limitations
Scalability:
The current implementation uses basic Python libraries like NumPy and is not optimized for very large datasets. For datasets with millions of rows or thousands of features, the execution time could be significantly higher compared to libraries like scikit-learn.

Non-Linearity:
The model assumes a linear relationship between features and the target variable. Non-linear relationships cannot be effectively modeled without additional feature transformations or kernel-based methods.

Sparse Datasets:
Although ElasticNet is designed for sparse datasets, handling extremely sparse or imbalanced datasets may require further preprocessing, such as imputation, normalization, or sampling techniques.

No Automated Parameter Tuning:
The implementation lacks features like cross-validation or grid search for automated hyperparameter tuning, which are commonly used to optimize performance.

### Future Work
Performance Optimization:
Enhance the computational efficiency by parallelizing gradient descent or leveraging GPU-based computation using libraries like CuPy or TensorFlow.

Expand Dataset Compatibility:
Add support for datasets with categorical variables through automated encoding techniques and preprocessing pipelines.

Hyperparameter Tuning:
Integrate automated hyperparameter tuning methods like grid search or random search to streamline experimentation.

Non-Linear Extensions:
Introduce polynomial features or kernel methods to capture non-linear relationships in the data.

Robustness Testing:
Conduct extensive tests with more diverse datasets, including high-dimensional, sparse, or noisy data, to further validate the model's effectiveness.

Visualization Enhancements:
Improve visualizations by including residual plots, regularization path plots, and feature importance rankings for better interpretability.

### Conclusion
The ElasticNet regression model implemented in this project demonstrates the ability to combine L1 and L2 regularization effectively, balancing feature selection and coefficient shrinkage. By building the model from scratch, we gained deeper insights into the inner workings of gradient descent, regularization, and optimization techniques. The model performed well on the California Housing dataset, producing comparable results to established libraries like scikit-learn.

While the implementation is robust for small to medium-sized datasets, challenges with scalability, non-linearity, and parameter tuning highlight opportunities for future improvements. With additional optimization and feature expansion, this implementation can serve as a solid foundation for more complex machine learning applications.



