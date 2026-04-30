import spacy

nlp = spacy.load("en_core_web_sm")

def extract_aspects(sentence):
    doc = nlp(sentence)
    aspects = set()   # use set to remove duplicates

    # Rule 1: adjectival modifier
    for token in doc:
        if token.dep_ == "amod" and token.head.pos_ == "NOUN":
            aspects.add((token.head.text, token.text))

    # Rule 2: noun + is + adjective
    for token in doc:
        if token.pos_ == "ADJ" and token.dep_ == "acomp":
            subj = [w for w in token.head.children if w.dep_ == "nsubj" and w.pos_ == "NOUN"]
            for s in subj:
                aspects.add((s.text, token.text))

    return list(aspects)

# TEST
reviews = [
    "The battery is bad but the screen is good",
    "Camera quality is excellent",
    "The phone heats up quickly"
]

for r in reviews:
    print("Review:", r)
    print("Aspects:", extract_aspects(r))
    print("-" * 40)
