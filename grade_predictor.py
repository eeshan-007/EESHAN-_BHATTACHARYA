import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
data = pd.read_csv("students.csv")
X = data[["StudyHours", "Attendance", "PreviousMarks"]]
y = data["FinalMarks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model Evaluation")
print("----------------")
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

print("\nEnter Student Details")

study_hours = float(input("Study Hours: "))
attendance = float(input("Attendance (%): "))
previous_marks = float(input("Previous Marks: "))

predicted_marks = model.predict(
    [[study_hours, attendance, previous_marks]]
)

print("\nPredicted Final Marks:", round(predicted_marks[0], 2))

marks = predicted_marks[0]

if marks >= 90:
    grade = "A+"
elif marks >= 80:
    grade = "A"
elif marks >= 70:
    grade = "B"
elif marks >= 60:
    grade = "C"
else:
    grade = "D"

print("Predicted Grade:", grade)