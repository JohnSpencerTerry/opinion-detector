import re
import spacy
from textblob import TextBlob

def is_uncited_statement(sentence):
    """Basic heuristic checks to flag opinionated or uncited statements."""
    opinion_keywords = ["designed to", "intended to", "meant to", "rigged", "corrupt", "biased", "control"]
    hedging_phrases = ["some say", "many believe", "it is thought", "people think", "it is said"]
    
    # Check for opinion keywords
    if any(keyword in sentence.lower() for keyword in opinion_keywords):
        return True
    
    # Check for hedging phrases
    if any(phrase in sentence.lower() for phrase in hedging_phrases):
        return True
    
    # Sentiment analysis (highly subjective sentences are flagged)
    sentiment = TextBlob(sentence).sentiment.polarity
    if sentiment > 0.5 or sentiment < -0.5:
        return True
    
    return False

def detect_uncited_statements(text):
    """Processes text and flags uncited statements."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    flagged_sentences = []
    for sent in doc.sents:
        if is_uncited_statement(sent.text):
            flagged_sentences.append(sent.text)
    
    return flagged_sentences

# Example usage
text = """
Waffles are made from a batter or dough cooked between two patterned plates, creating a crispy texture.
Waffles are hands down the best breakfast food, with their perfect balance of crispy edges and soft, airy inside—nothing else even comes close!
"""

flagged = detect_uncited_statements(text)
print("Flagged Statements:")
for sentence in flagged:
    print(f"- {sentence}")
