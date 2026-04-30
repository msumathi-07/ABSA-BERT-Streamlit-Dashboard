import pandas as pd
import spacy


nlp = spacy.load("en_core_web_sm")


POSITIVE_WORDS = {
    "good", "excellent", "great", "awesome",
    "nice", "amazing", "fantastic", "perfect"
}

NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "worst",
    "awful", "slow", "disappointing"
}

def get_sentiment(opinion):
    op = opinion.lower()
    if op in POSITIVE_WORDS:
        return "Positive"
    elif op in NEGATIVE_WORDS:
        return "Negative"
    else:
        return "Neutral"

def extract_aspects(sentence):
    doc = nlp(sentence)
    aspects = set()

    for token in doc:
        if (
            token.dep_ == "amod"
            and token.pos_ == "ADJ"
            and token.head.pos_ == "NOUN"
        ):
            aspects.add((token.head.text.lower(), token.text.lower()))

    for token in doc:
        if token.dep_ == "acomp" and token.pos_ == "ADJ":
            for child in token.head.children:
                if child.dep_ == "nsubj" and child.pos_ == "NOUN":
                    aspects.add((child.text.lower(), token.text.lower()))

    return list(aspects)

df = pd.read_csv("dataset.csv")   

df["Sentence"] = df["Sentence"].astype(str).str.lower()

results = []

for _, row in df.iterrows():
    sentence = row["Sentence"]
    extracted = extract_aspects(sentence)

    for aspect, opinion in extracted:
        sentiment = get_sentiment(opinion)
        results.append({
            "Sentence": sentence,
            "Aspect": aspect,
            "Opinion": opinion,
            "Sentiment": sentiment
        })

result_df = pd.DataFrame(results)

print("\nSample Output:")
print(result_df.head(10))

result_df.to_csv("absa_output.csv", index=False)

print("\nABSA completed. Output saved as absa_output.csv")