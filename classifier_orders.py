import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score

df = pd.read_excel("Dataset_for_Data_Analytics.ods", engine="odf")

df["CouponCode"] = df["CouponCode"].fillna("NoCoupon")
drop_cols = ["OrderID", "TrackingNumber", "CustomerID", "ShippingAddress", "Date"]
data = df.drop(columns=drop_cols)

target_col = "OrderStatus"
categorical_cols = ["Product", "PaymentMethod", "CouponCode", "ReferralSource"]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

target_encoder = LabelEncoder()
data[target_col] = target_encoder.fit_transform(data[target_col])

X = data.drop(columns=[target_col])
y = data[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, target_names=target_encoder.classes_))
print("F1:", f1_score(y_test, predictions, average='weighted'))