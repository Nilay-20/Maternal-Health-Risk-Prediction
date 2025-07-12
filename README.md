<h1>Maternal Health Risk Prediction</h1>

<p>A machine learning project to predict maternal health risk levels (Low, Mid, High) based on physiological data using classification models.</p>

<h2>Overview</h2>
<p>
  This project uses a dataset from the 
  <a href="https://archive.ics.uci.edu/dataset/863/maternal+health+risk" target="_blank">UCI Machine Learning Repository</a>. 
  It applies <strong>K-Nearest Neighbors</strong> and <strong>Random Forest</strong> to classify maternal risk based on:
</p>
<ul>
  <li>Age</li>
  <li>Diastolic Blood Pressure</li>
  <li>Blood Sugar</li>
  <li>Body Temperature</li>
  <li>Heart Rate</li>
</ul>

<h2>Tech Stack</h2>
<ul>
  <li>Python (Jupyter Notebook)</li>
  <li>Libraries: pandas, numpy, scikit-learn, seaborn, matplotlib</li>
</ul>

<h2>Results</h2>
<ul>
  <li><strong>KNN Accuracy:</strong> 83.7%</li>
  <li><strong>Random Forest Accuracy:</strong> 83.74%</li>
  <li>Random Forest outperforms due to ensemble learning and better feature handling</li>
</ul>
<p>To run this project clone the repository and run the app.py file using streamlit run app.py</p>

<h2>Highlights</h2>
<ul>
  <li>Preprocessing, EDA, Model Training, Cross-Validation</li>
  <li>RF provides better consistency and lower RMSE</li>
  <li>Supports early identification of high-risk pregnancies</li>
</ul>
