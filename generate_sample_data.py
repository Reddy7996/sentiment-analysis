"""
Generate Sample Movie Review Dataset
Creates a realistic reviews.csv file for the sentiment analysis project
"""

import pandas as pd
import numpy as np
import random
import os

# Set random seed
np.random.seed(42)
random.seed(42)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Positive review templates
positive_templates = [
    "This movie was {adj1} and {adj2}! I {verb} it so much.",
    "{adj1} film with {adj2} performances. Highly recommend!",
    "Absolutely {adj1}! The {noun} was {adj2}.",
    "I {verb} every minute. {adj1} and {adj2} throughout.",
    "{adj1} masterpiece. The {noun} and {noun2} were {adj2}.",
    "What a {adj1} experience! {adj2} from start to finish.",
    "This is {adj1}! {adj2} {noun} and brilliant {noun2}.",
    "{verb} this movie! {adj1}, {adj2}, and unforgettable.",
    "The {noun} was {adj1} and the {noun2} was {adj2}. Perfection!",
    "{adj1} and {adj2}! A must-watch film for everyone."
]

# Negative review templates
negative_templates = [
    "This movie was {adj1} and {adj2}. Complete waste of time.",
    "{adj1} film with {adj2} {noun}. Do not recommend.",
    "Absolutely {adj1}! The {noun} was {adj2}.",
    "I {verb} every minute. {adj1} and {adj2} throughout.",
    "{adj1} disaster. The {noun} and {noun2} were {adj2}.",
    "What a {adj1} experience! {adj2} from start to finish.",
    "This is {adj1}! {adj2} {noun} and terrible {noun2}.",
    "{verb} this movie! {adj1}, {adj2}, and forgettable.",
    "The {noun} was {adj1} and the {noun2} was {adj2}. Awful!",
    "{adj1} and {adj2}! A complete disappointment."
]

# Word banks
positive_adjectives = [
    'amazing', 'fantastic', 'wonderful', 'brilliant', 'excellent',
    'outstanding', 'superb', 'incredible', 'spectacular', 'magnificent',
    'stunning', 'beautiful', 'perfect', 'awesome', 'great',
    'marvelous', 'exceptional', 'phenomenal', 'remarkable', 'splendid'
]

negative_adjectives = [
    'terrible', 'awful', 'horrible', 'boring', 'dull',
    'disappointing', 'poor', 'bad', 'worst', 'mediocre',
    'tedious', 'unwatchable', 'bland', 'forgettable', 'weak',
    'pathetic', 'lousy', 'abysmal', 'disastrous', 'atrocious'
]

positive_verbs = [
    'loved', 'enjoyed', 'adored', 'treasured', 'appreciated',
    'cherished', 'admired', 'relished'
]

negative_verbs = [
    'hated', 'despised', 'detested', 'regretted', 'suffered through',
    'endured', 'disliked'
]

nouns = [
    'acting', 'plot', 'cinematography', 'direction', 'screenplay',
    'story', 'script', 'pacing', 'dialogue', 'characters',
    'visuals', 'soundtrack', 'editing', 'performances', 'cast'
]

def generate_review(sentiment, template, idx):
    """Generate a single review"""
    
    if sentiment == 1:  # Positive
        adj1 = random.choice(positive_adjectives)
        adj2 = random.choice(positive_adjectives)
        while adj2 == adj1:
            adj2 = random.choice(positive_adjectives)
        verb = random.choice(positive_verbs)
    else:  # Negative
        adj1 = random.choice(negative_adjectives)
        adj2 = random.choice(negative_adjectives)
        while adj2 == adj1:
            adj2 = random.choice(negative_adjectives)
        verb = random.choice(negative_verbs)
    
    noun = random.choice(nouns)
    noun2 = random.choice(nouns)
    while noun2 == noun:
        noun2 = random.choice(nouns)
    
    review = template.format(
        adj1=adj1,
        adj2=adj2,
        verb=verb,
        noun=noun,
        noun2=noun2
    )
    
    # Add some variation
    if random.random() < 0.3:
        exclamations = ['!', '!!', '!!!']
        review = review.replace('.', random.choice(exclamations))
    
    return review

# Generate dataset
reviews = []
sentiments = []
n_samples = 5000

print(f"Generating {n_samples} movie reviews...")

for i in range(n_samples // 2):
    # Positive review
    template = random.choice(positive_templates)
    review = generate_review(1, template, i)
    reviews.append(review)
    sentiments.append(1)
    
    # Negative review
    template = random.choice(negative_templates)
    review = generate_review(0, template, i)
    reviews.append(review)
    sentiments.append(0)

# Create DataFrame
df = pd.DataFrame({
    'review': reviews,
    'sentiment': sentiments
})

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add review ID
df.insert(0, 'id', range(1, len(df) + 1))

# Save to CSV
df.to_csv('data/reviews.csv', index=False)

print(f"\n✓ Dataset created successfully!")
print(f"\nDataset Statistics:")
print(f"- Total reviews: {len(df):,}")
print(f"- Positive reviews: {(df['sentiment'] == 1).sum():,} ({(df['sentiment'] == 1).sum()/len(df)*100:.1f}%)")
print(f"- Negative reviews: {(df['sentiment'] == 0).sum():,} ({(df['sentiment'] == 0).sum()/len(df)*100:.1f}%)")
print(f"\nSample reviews:")
print("\nPositive:")
print(df[df['sentiment'] == 1].sample(3)['review'].values)
print("\nNegative:")
print(df[df['sentiment'] == 0].sample(3)['review'].values)
print(f"\n✓ Saved to: data/reviews.csv")
