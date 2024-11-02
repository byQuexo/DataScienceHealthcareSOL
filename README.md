# MachineLearningSOL

## Building a Forecast Model for Healthcare Data

### Objectives of this Notebook:

- **Data Structuring**: Organizing the healthcare dataset into a suitable format for analysis and modeling.
- **Data Cleaning**: Handling missing values, correcting inconsistencies, and preparing the data for use.
- **Data Mining and Visualization**: Extracting insights from the data using visualizations and exploratory data analysis techniques.
- **Model Training**: Developing and evaluating machine learning models to forecast healthcare-related outcomes.

### Learning Material:
- **Pandas DataFrame operation**:
    - Pandas Documentation: https://pandas.pydata.org/docs/
    - DataFrame Indexing: https://pandas.pydata.org/docs/user_guide/indexing.html
- **Matplotlib for data visualization**:
    - Matplotlib Documentation: https://matplotlib.org/stable/contents.html
    - Creating subplots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
- **NumPy for numerical operations**: 
    - NumPy Documentation: https://numpy.org/doc/stable/
    - NumPy arrange function: https://numpy.org/doc/stable/reference/generated/numpy.arange.html
- **Data visualization techniques**:
    - Grouped bar charts: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    - Customizing plots: https://matplotlib.org/stable/tutorials/introductory/customizing.html
- **scikit-learn Documentation**:
   - Main page: https://scikit-learn.org/stable/
   - RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
   - GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- **Model Evaluation**:
   - Classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
   - accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
   - classification_report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- **Data Splitting**:
   - train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- **Joblib for model persistence**:
   - Joblib documentation: https://joblib.readthedocs.io/en/latest/
   - Persisting scikit-learn models: https://scikit-learn.org/stable/model_persistence.html
- **IPython Display**:
   - IPython display module: https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html
- **Feature Engineering**:
   - Pandas get_dummies for one-hot encoding: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
- **Python Standard Library**:
   - Exception handling: https://docs.python.org/3/tutorial/errors.html

[1] https://pandas.pydata.org/docs/
[2] https://pandas.pydata.org/docs/user_guide/indexing.html

> [!NOTE]
> 
> Dataset: [National Health and Nutrition Examination Survey (NHANES) – Vision and Eye Health Surveillance](https://healthdata.gov/dataset/National-Health-and-Nutrition-Examination-Survey-N/mbgv-hccf/about_data)
> This dataset is sourced from [Centers for Disease Control and Prevention](https://www.cdc.gov/visionhealth/vehss/index.html).

