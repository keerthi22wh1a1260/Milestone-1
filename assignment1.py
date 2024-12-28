import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"  #pre-trained emotion classification model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
def classify_emotion(input_text):
    input_tokens = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    output = model(**input_tokens)
    prediction = torch.argmax(output.logits, dim=1).item()
    emotions = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
    emotion = emotions[prediction]
    return emotion
input_text = input("Enter a sentence to classify emotion: ").strip()
if input_text:  
    
    emotion = classify_emotion(input_text)
    print("Predicted Emotion:", emotion)
else:
    print("Input text is empty. Please enter a valid sentence.")
