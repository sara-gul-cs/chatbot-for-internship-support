# Chatbot for Internship Support

Objective :
Develop an AI chatbot to answer intern queries and provide support,
automating real-time responses based on FAQ content.

Dataset :
15 FAQ question-answer pairs covering common internship queries: duration,
task submission, tools needed, certificates, support contact, deadlines, etc.

Approach :
1. **TF-IDF Vectorization** — converted all known FAQ questions into numeric
   vectors (same technique used in the Sentiment Analysis and Skill Gap
   Analysis tasks).
2. **Cosine Similarity Matching** — when a user asks a question, it's
   vectorized the same way and compared against all known FAQ questions to
   find the closest match.
3. **Confidence Threshold (new concept)** — if the best match's similarity
   score is below 0.2, the bot admits it doesn't know rather than returning
   a poor guess. This is what makes it behave like a real support bot
   instead of always forcing an answer.

## Results
Tested on 4 sample queries:

| User Query | Confidence | Bot Response |
|---|---|---|
| "How long does the internship last?" | 1.00 | Correctly matched: internship duration |
| "Where do I send my finished project?" | 0.00 | **Fallback triggered** (see limitation below) |
| "Will I get paid for this?" | 0.76 | Correctly matched: compensation info |
| "What's the weather like today?" | 0.00 | Correctly triggered fallback (unrelated query) |

Limitation : 
The second query — "Where do I send my finished project?" — should have
matched the FAQ about submitting work, but got 0.00 confidence and
triggered the fallback instead. This happened because TF-IDF only matches
literal keywords: "send/finished/project" share no words with the FAQ's
phrasing ("submit/work"). This is a well-known limitation of purely
keyword-based matching — it has no understanding of synonyms or paraphrasing.

**To fix this in a production version:**
- Use semantic embeddings (e.g. Sentence-BERT or OpenAI embeddings) instead
  of TF-IDF, which understand meaning rather than exact word overlap
- Expand the FAQ with paraphrased versions of each question
- Add a simple synonym-expansion step before vectorizing

This limitation is intentional to include in the report — it demonstrates
understanding of *why* the chatbot fails on certain phrasing, not just that
it works on others.

## Files
- `faq_data.csv` — the FAQ question-answer dataset
- `chatbot.py` — full pipeline (TF-IDF → similarity matching → confidence
  threshold → demo queries). An interactive chat loop is included but
  commented out at the bottom of the file - uncomment it to chat live.

## How to run
```
pip install pandas scikit-learn
python3 chatbot.py
```

To chat with it live instead of just running the demo, open `chatbot.py`,
uncomment the "INTERACTIVE MODE" section at the bottom, and run again.
