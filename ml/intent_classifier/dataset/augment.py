import pandas as pd
import random

df = pd.read_csv("raw_intents.csv")



def add_filler(text):
    fillers = ["please", "can you", "i want to", "could you", "just", "hey"]
    word = random.choice(fillers)
    return f"{word} {text}"

def add_suffix(text):
    suffixes = ["for me", "right now", "please", "quickly", "now"]
    word = random.choice(suffixes)
    return f"{text} {word}"

def swap_words(text):
    words = text.split()
    if len(words) < 3:
        return text
    i, j = random.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)

def remove_word(text):
    words = text.split()
    if len(words) <= 2:
        return text
    words.pop(random.randint(0, len(words)-1))
    return " ".join(words)

augment_funcs = [add_filler, add_suffix, swap_words, remove_word]


augmented_rows = []

for _, row in df.iterrows():
    text = row["text"]
    intent = row["intent"]
    
    
    augmented_rows.append({"text": text, "intent": intent})
    
    
    for _ in range(3):
        func = random.choice(augment_funcs)
        new_text = func(text)
        augmented_rows.append({"text": new_text, "intent": intent})

augmented_df = pd.DataFrame(augmented_rows)
augmented_df = augmented_df.drop_duplicates(subset=["text"])
augmented_df.to_csv("augmented_intents.csv", index=False)

print(f"Original samples : {len(df)}")
print(f"Augmented samples: {len(augmented_df)}")