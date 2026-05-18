import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression

# Dataset
data = {
    "Study Hours": [1,2,3,4,5,6,7,8,9,10],
    "Pass/Fail": [0,0,0,0,1,1,1,1,1,1]
}
df = pd.DataFrame(data)

# Train model
x = df["Study Hours"].values.reshape(-1,1)
y = df["Pass/Fail"]

model = LogisticRegression()
model.fit(x, y)

st.title("Logistic Regression - Study Hours vs Pass/Fail")

st.write("Dataset")
st.dataframe(df)

# Input
hours = st.slider("Enter Study Hours", 1, 10, 5)

# Prediction
z = model.coef_[0][0]*hours + model.intercept_[0]
prob = 1/(1+math.exp(-z))

if prob >= 0.5:
    result = "Pass"
else:
    result = "Fail"

st.subheader("Prediction")
st.write("Probability:", prob)
st.write("Result:", result)

# Sigmoid curve
x_vals = np.linspace(-10,10,100)
y_vals = 1/(1+np.exp(-x_vals))

fig, ax = plt.subplots()
ax.plot(x_vals, y_vals)
ax.set_title("Sigmoid Function")
st.pyplot(fig)