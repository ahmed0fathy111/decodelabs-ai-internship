import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_excel("Dataset_for_Data_Analytics.ods", engine="odf")

category_map = {
    "Monitor": "Electronics", "Phone": "Electronics", "Tablet": "Electronics",
    "Laptop": "Electronics", "Printer": "Electronics",
    "Chair": "Furniture", "Desk": "Furniture",
}

catalog = df.copy()
catalog["Category"] = catalog["Product"].map(category_map)

low_cut, high_cut = df["UnitPrice"].quantile([0.33, 0.67])

def price_tier(price):
    if price <= low_cut:
        return "cheap budget affordable"
    elif price <= high_cut:
        return "midrange moderate"
    else:
        return "premium expensive high-end"

catalog["PriceTier"] = catalog["UnitPrice"].apply(price_tier)

items = (
    catalog.groupby(["Product", "Category", "PriceTier"])["UnitPrice"]
    .mean()
    .reset_index()
)
items["tags"] = (items["Product"] + " " + items["Category"] + " " + items["PriceTier"]).str.lower()

vectorizer = TfidfVectorizer()
item_vectors = vectorizer.fit_transform(items["tags"])

def recommend(user_query, top_n=3):
    query_vector = vectorizer.transform([user_query.lower()])
    scores = cosine_similarity(query_vector, item_vectors).flatten()
    ranked = items.copy()
    ranked["Score"] = scores
    ranked = ranked.sort_values("Score", ascending=False)
    return ranked.head(top_n)[["Product", "Category", "PriceTier", "UnitPrice", "Score"]]

# Interactive loop
print("Type preferences like 'cheap electronics'. Type 'exit' to quit.\n")
while True:
    user_input = input("Your preferences: ").strip()
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break
    if not user_input:
        continue
    print(recommend(user_input, top_n=3).to_string(index=False))
    print()