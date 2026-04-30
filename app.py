import streamlit as st
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------- SETTINGS -------------------- #
st.set_page_config(page_title="ABSA Dashboard", layout="wide")
sns.set(style="whitegrid", palette="Set2")

# -------------------- LOAD BERT MODEL -------------------- #
@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("absa_model")
    model = AutoModelForSequenceClassification.from_pretrained("absa_model")
    return tokenizer, model

tokenizer, model = load_bert()

# -------------------- SENTIMENT FUNCTION -------------------- #
def predict_sentiment(sentence, aspect):
    
    text = sentence + " [SEP] " + aspect

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()

    if pred == 0:
        return "Negative"
    elif pred == 1:
        return "Neutral"
    else:
        return "Positive"
# -------------------- LOAD SPACY -------------------- #
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# -------------------- ASPECT EXTRACTION -------------------- #
def extract_aspects(sentence):
    doc = nlp(sentence)
    aspects = set()

    for token in doc:
        if token.dep_ == "amod" and token.pos_ == "ADJ" and token.head.pos_ == "NOUN":

            aspects.add((token.head.text.lower(), token.text.lower()))

    for token in doc:
        if token.dep_ == "acomp" and token.pos_ == "ADJ":
            for child in token.head.children:
                if child.dep_ == "nsubj" and child.pos_ == "NOUN":
                    aspects.add((child.text.lower(), token.text.lower()))

    return list(aspects)

# -------------------- UI -------------------- #
st.title("📊 Aspect-Based Sentiment Analysis Dashboard")
st.markdown("Identify **defective product features** from customer reviews.")

mode = st.radio("Choose Analysis Mode:",
                ["Single Review", "Multiple Reviews (CSV Upload)"])

# ==========================================================
# ================= SINGLE REVIEW MODE =====================
# ==========================================================
if mode == "Single Review":

    review = st.text_area("✍️ Enter Customer Review:")

    if st.button("🔍 Analyze Review"):

        extracted = extract_aspects(review)

        if not extracted:
            st.warning("No aspects detected. Try a more descriptive review.")
        else:
            results = []

            # Split review into smaller clauses
            import re
            clauses = re.split(r'\band\b|\bbut\b|\bwhile\b|,', review, flags=re.I)

            for aspect, _ in extracted:
                aspect_found = False
                for clause in clauses:
                    if aspect.lower() in clause.lower():
                        # feed only the clause containing the aspect
                        sentiment = predict_sentiment(clause, aspect)
                        results.append({
                            "Aspect": aspect,
                            "Sentiment": sentiment.capitalize()  # Positive / Negative / Neutral
                        })
                        aspect_found = True
                        break
                # fallback if aspect not found in any clause
                if not aspect_found:
                    sentiment = predict_sentiment(review, aspect)
                    results.append({
                        "Aspect": aspect,
                        "Sentiment": sentiment.capitalize()
                    })

            # Create DataFrame
            df = pd.DataFrame(results)

            # Metrics
            total_aspects = len(df)
            positive_count = (df["Sentiment"] == "Positive").sum()
            negative_count = (df["Sentiment"] == "Negative").sum()
            neutral_count = (df["Sentiment"] == "Neutral").sum()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Aspects", total_aspects)
            col2.metric("Positive", positive_count)
            col3.metric("Negative", negative_count)
            col4.metric("Neutral", neutral_count)

            st.dataframe(df, use_container_width=True)
# ==========================================================
# ================= MULTI REVIEW MODE ======================
# ==========================================================
else:

    uploaded_file = st.file_uploader("Upload CSV File with 'review' column", type=["csv"])

    if uploaded_file is not None:

        df_reviews = pd.read_csv(uploaded_file)

        if "review" not in df_reviews.columns:
            st.error("CSV must contain a column named 'review'")
        else:

            results = []

            for review in df_reviews["review"]:
                extracted = extract_aspects(str(review))

                for aspect, opinion in extracted:
                    sentiment = predict_sentiment(review, aspect)
                    results.append({
                        "Aspect": aspect,
                        "Sentiment": sentiment
                    })

            if len(results) == 0:
                st.warning("No aspects detected in dataset.")
            else:
                result_df = pd.DataFrame(results)

                summary = result_df.groupby(
                    ["Aspect", "Sentiment"]
                ).size().unstack(fill_value=0)

                summary["Total"] = summary.sum(axis=1)

                if "Positive" in summary.columns and "Negative" in summary.columns:
                    summary["NPS"] = ((summary["Positive"] - summary["Negative"]) / summary["Total"]) * 100

                st.subheader("📊 Aspect Score (NPS)")
                st.dataframe(summary)

                # -------- NEGATIVE --------
                if "Negative" in summary.columns:
                    neg_counts = summary["Negative"].sort_values(ascending=False)

                    st.subheader("🔴 Most Problematic Aspects")

                    fig1, ax1 = plt.subplots()
                    sns.barplot(x=neg_counts.values, y=neg_counts.index, ax=ax1)
                    st.pyplot(fig1)

                    st.success(f"🚨 Most Defective Aspect: **{neg_counts.idxmax()}**")

                # -------- POSITIVE --------
                if "Positive" in summary.columns:
                    pos_counts = summary["Positive"].sort_values(ascending=False)

                    st.subheader("🟢 Strongest Product Features")

                    fig2, ax2 = plt.subplots()
                    sns.barplot(x=pos_counts.values, y=pos_counts.index, ax=ax2)
                    st.pyplot(fig2)

                    st.info(f"⭐ Best Performing Aspect: **{pos_counts.idxmax()}**")

                # -------- RADAR --------
                st.subheader("📡 Strength vs Weakness Radar")

                top_aspects = summary.sort_values("Total", ascending=False).head(5)

                labels = top_aspects.index.tolist()
                values = top_aspects["NPS"].fillna(0).tolist()

                angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
                values += values[:1]
                angles = np.concatenate((angles, [angles[0]]))

                fig3 = plt.figure()
                ax3 = plt.subplot(111, polar=True)

                ax3.plot(angles, values)
                ax3.fill(angles, values, alpha=0.3)

                ax3.set_xticks(angles[:-1])
                ax3.set_xticklabels(labels)

                st.pyplot(fig3)

                # -------- RECOMMEND --------
                st.subheader("💡 Product Improvement Suggestions")

                for aspect in summary.index:
                    if "NPS" in summary.columns:
                        score = summary.loc[aspect, "NPS"]

                        if score < 0:
                            st.error(f"🔴 Improve {aspect}")
                        elif score > 50:
                            st.success(f"🟢 Maintain {aspect}")
                        else:
                            st.warning(f"🟡 Monitor {aspect}")