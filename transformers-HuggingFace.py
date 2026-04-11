from transformers import AutoTokenizer , pipeline
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
Tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis",model_name,tokenizer=Tokenizer)
def predict_sentiment(review):
    result = classifier(review)[0]
    label = result["label"].lower()
    score = result["score"]
    if label == "positive":
        print(f"Positive with confidence: {score:.0%}")
    elif label =="negative":
        print(f"Negative with confidence: {score:.0%}")
    elif label == "neutral":
        print(f"neutral with confidence: {score:.0%}")
predict_sentiment("formula 1 was a very weird movie. On one hand, it was cool and exhilarating , but i also felt it was rushed and chaotic at the same time")




