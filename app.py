
import numpy as np
import pandas as pd 
import io 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.model_selection import KFold
import streamlit as st


df = pd.read_csv('Maternal Health Risk Data Set.csv')


st.title("Maternal Health Risk Classification")

st.subheader("Head of the Dataset:")
st.dataframe(df.head())

st.subheader("Info about the Dataset:")
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()

st.subheader("DataFrame Info")
st.text(info_str)

st.write("The dataset size:", df.shape)

st.subheader("Risk Level Value Counts:")
st.write(df["RiskLevel"].value_counts())

st.subheader("Tail of the Dataset:")
st.dataframe(df.tail())

st.subheader("Statistical Summary:")
st.dataframe(df.describe().T)


st.subheader("Boxplot for Outlier Detection")
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 15))
for ax, column in zip(axes.flatten(), df.columns):
    sns.boxplot(y=df[column], color="#4682B4", ax=ax)
    ax.set_title(f"{column}", fontsize=18)
plt.tight_layout()
st.pyplot(fig)

risk_mapping = {"low risk": 0, "mid risk": 1, "high risk": 2}
df["RiskLevel"] = df["RiskLevel"].map(risk_mapping)

st.subheader("Correlation Heatmap")
plt.figure(figsize=(22, 20))
sns.heatmap(df.corr(), annot=True, cmap="GnBu")
plt.title("Correlation Heatmap of Variables", fontsize=16)
st.pyplot(plt.gcf())


df = df.drop(["SystolicBP"], axis=1)


df = df.drop(df.index[df.HeartRate == 7])


columns = ["Age", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
scale_X = StandardScaler()
X = pd.DataFrame(scale_X.fit_transform(df.drop(["RiskLevel"], axis=1)), columns=columns)
y = df["RiskLevel"]
st.subheader("First 5 rows of feature data (X):")
st.dataframe(X.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


knn = KNeighborsClassifier(n_neighbors=10, p=2, weights="distance")
knn_mod = knn.fit(X_train, y_train)
pred_knn = knn_mod.predict(X_test)

mse_knn = mean_squared_error(y_test, pred_knn)
rmse_knn = np.sqrt(mse_knn)
acc = accuracy_score(pred_knn, y_test)

st.subheader("K-Nearest Neighbors Model Evaluation")
st.write("Mean Square Error for KNN =", mse_knn)
st.write("Root Mean Square Error for KNN =", rmse_knn)
st.write("Accuracy for KNN model = ", acc)

st.text("Classification Report")
st.text(classification_report(y_test, pred_knn))
st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, pred_knn))

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, pred_knn), annot=True, ax=ax, cmap="GnBu")
ax.set_xlabel("Predicted Risk Levels")
ax.set_ylabel("True Risk Levels")
ax.set_title("Confusion Matrix") 
ax.xaxis.set_ticklabels(["Low", "Mid", "High"])
ax.yaxis.set_ticklabels(["Low", "Mid", "High"])
st.pyplot(fig)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(knn, X, y, cv=kf)
st.write("Cross Validation Scores: ", scores)
st.write("Average CV Score: ", scores.mean())

y_pred = cross_val_predict(knn, X, y, cv=kf)


random_forest = RandomForestClassifier(criterion="entropy", max_depth=30, min_samples_leaf=1, min_samples_split=2, n_estimators=200, random_state=42)
random_forest_mod = random_forest.fit(X_train, y_train)
pred_random_forest = random_forest_mod.predict(X_test)

mse_random_forest = mean_squared_error(y_test, pred_random_forest)
rmse_random_forest = np.sqrt(mse_random_forest)
acc = accuracy_score(pred_random_forest, y_test)

st.subheader("Random Forest Model Evaluation")
st.write("Mean Square Error for Random Forest =", mse_random_forest)
st.write("Root Mean Square Error for Random Forest =", rmse_random_forest)
st.write("Accuracy of Random Forest Model = ", acc)

st.text("Classification Report")
st.text(classification_report(y_test, pred_random_forest))
st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, pred_random_forest))

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, pred_random_forest), annot=True, ax=ax, cmap="GnBu")
ax.set_xlabel("Predicted Risk Levels")
ax.set_ylabel("True Risk Levels")
ax.set_title("Confusion Matrix") 
ax.xaxis.set_ticklabels(["Low", "Mid", "High"])
ax.yaxis.set_ticklabels(["Low", "Mid", "High"])
st.pyplot(fig)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(random_forest, X, y, cv=kf)
st.write("Cross Validation Scores: ", scores)
st.write("Average CV Score: ", scores.mean())

y_pred = cross_val_predict(random_forest, X, y, cv=kf)


model_results = pd.DataFrame(columns=['Model', 'Accuracy', 'MSE', 'RMSE', 'CV Score'])


models = [
    ('KNN', knn, pred_knn),
    ('Random Forest', random_forest, pred_random_forest)
]


for model_name, model, preds in models:
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    accuracy = accuracy_score(y_test, preds)
    cv_score = cross_val_score(model, X, y, cv=kf).mean()
    
    temp_df = pd.DataFrame([{
        'Model': model_name,
        'Accuracy': accuracy,
        'MSE': mse,
        'RMSE': rmse,
        'CV Score': cv_score
    }])
    
    model_results = pd.concat([model_results, temp_df], ignore_index=True)

st.subheader("Model Evaluation Summary:")
st.dataframe(model_results)


st.header("🔍 Predict Maternal Health Risk")

with st.form("user_input_form"):
    st.subheader("Enter Patient Data:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=30)
        diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=140, value=70)
    with col2:
        bs = st.number_input("Blood Sugar (BS)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
        body_temp = st.number_input("Body Temperature (°F)", min_value=90.0, max_value=110.0, value=98.6, step=0.1)
    with col3:
        heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200, value=80)

    model_choice = st.selectbox("Choose a model for prediction", 
                                ["K-Nearest Neighbors", "Random Forest"])

    submit_button = st.form_submit_button("Predict")

if submit_button:
    user_data = pd.DataFrame([[age, diastolic_bp, bs, body_temp, heart_rate]],
                             columns=["Age", "DiastolicBP", "BS", "BodyTemp", "HeartRate"])
    scaled_user_data = scale_X.transform(user_data)

    if model_choice == "K-Nearest Neighbors":
        model = knn_mod
    else:
        model = random_forest_mod

    prediction = model.predict(scaled_user_data)[0]
    prediction_proba = model.predict_proba(scaled_user_data)[0]

    risk_label = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}[prediction]
    risk_color = {0: "🟢", 1: "🟠", 2: "🔴"}[prediction]

    st.subheader("Prediction Result")
    st.markdown(f"### {risk_color} Predicted Risk Level: **{risk_label}**")

    confidence = round(prediction_proba[prediction] * 100, 2)
    st.markdown(f"**Model Confidence:** {confidence}%")
