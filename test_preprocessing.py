from src.preprocess import preprocess_text


text = "I DON'T know how to cancel my order!!!"

clean_text = preprocess_text(text)

print("Original text:")
print(text)

print("\nProcessed text:")
print(clean_text)