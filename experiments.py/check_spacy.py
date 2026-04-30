import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The battery is bad but the screen is good")

for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)
