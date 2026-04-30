import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn style
sns.set(style="whitegrid")

# 1. Load ABSA output

df = pd.read_csv("absa_output.csv")
print("Original shape:", df.shape)

# 2. Remove duplicates

df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

# 3. Aspect Frequency Analysis

aspect_freq = df["Aspect"].value_counts().reset_index()
aspect_freq.columns = ["Aspect", "Frequency"]
aspect_freq.to_csv("aspect_frequency.csv", index=False)

# 4. Sentiment Distribution

sentiment_dist = df["Sentiment"].value_counts().reset_index()
sentiment_dist.columns = ["Sentiment", "Count"]
sentiment_dist.to_csv("sentiment_distribution.csv", index=False)

# 5. Aspect vs Sentiment Analysis
aspect_sentiment = (
    df.groupby(["Aspect", "Sentiment"])
    .size()
    .reset_index(name="Count")
)
aspect_sentiment.to_csv("aspect_sentiment_analysis.csv", index=False)

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sentiment", order=["Positive", "Neutral", "Negative"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

top_aspects = df["Aspect"].value_counts().head(10).reset_index()
top_aspects.columns = ["Aspect", "Frequency"]

plt.figure(figsize=(8, 4))
sns.barplot(data=top_aspects, x="Aspect", y="Frequency")
plt.title("Top 10 Frequently Mentioned Aspects")
plt.xlabel("Aspect")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

top5_aspects = (
    df["Aspect"].value_counts().head(5).index
)

filtered_df = df[df["Aspect"].isin(top5_aspects)]

plt.figure(figsize=(9, 5))
sns.countplot(
    data=filtered_df,
    x="Aspect",
    hue="Sentiment",
    order=top5_aspects
)
plt.title("Aspect-wise Sentiment Comparison (Top 5 Aspects)")
plt.xlabel("Aspect")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("Analysis and visualization completed successfully.")