import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1. LOAD FAQ DATA
# ============================================================
df = pd.read_csv('faq_data.csv')
print(f"Loaded {len(df)} FAQ entries.\n")

# ============================================================
# 2. TF-IDF VECTORIZATION OF ALL KNOWN QUESTIONS
# ============================================================
vectorizer = TfidfVectorizer(stop_words='english')
question_vectors = vectorizer.fit_transform(df['question'])

# ============================================================
# 3. CORE CHATBOT LOGIC
# ============================================================
# CONFIDENCE_THRESHOLD: if the best match is below this similarity score,
# the bot admits it doesn't know rather than guessing with a bad answer.
# This is the one genuinely new concept in this task.
CONFIDENCE_THRESHOLD = 0.2

def get_bot_response(user_query):
    user_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vector, question_vectors)[0]

    best_match_idx = similarities.argmax()
    best_score = similarities[best_match_idx]

    if best_score < CONFIDENCE_THRESHOLD:
        return ("I'm not sure about that one - please contact your mentor "
                "or check the Guidelines tab for more details."), best_score
    else:
        return df.iloc[best_match_idx]['answer'], best_score

# ============================================================
# 4. DEMO: TEST ON SAMPLE QUERIES (matches what to show in your video)
# ============================================================
test_queries = [
    "How long does the internship last?",          # close match to FAQ #1
    "Where do I send my finished project?",         # close match to FAQ #3
    "Will I get paid for this?",                    # close match to FAQ #10
    "What's the weather like today?",               # should trigger fallback
]

print("="*60)
print("CHATBOT DEMO - Sample Queries")
print("="*60)
for query in test_queries:
    response, score = get_bot_response(query)
    print(f"\nUser: {query}")
    print(f"Bot ({score:.2f} confidence): {response}")

# ============================================================
# 5. INTERACTIVE MODE (uncomment to chat live in your terminal)
# ============================================================
# print("\n" + "="*60)
# print("Interactive mode - type 'quit' to exit")
# print("="*60)
# while True:
#     user_input = input("\nYou: ")
#     if user_input.lower() == 'quit':
#         break
#     response, score = get_bot_response(user_input)
#     print(f"Bot: {response}")
