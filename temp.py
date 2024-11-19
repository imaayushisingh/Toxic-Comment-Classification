def classify_comment(comment):
    """
    Tokenize the input comment and predict confidence scores for all classes.
    """
    # Tokenize the comment
    inputs = tokenizer.encode_plus(
        comment,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )
    input_ids = inputs["input_ids"]
    
    # Predict using the model
    prediction = model.predict(input_ids)
    
    # Map predictions to labels
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    results = {labels[i]: float(prediction[0][i]) for i in range(len(labels))}
    
    return results

if __name__ == "__main__":
    print("Youtube Comment Multi-Label Classification")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("Enter a comment: ")
        if user_input.lower() == "exit":
            break
        results = classify_comment(user_input)
        print("\nClassification Results:")
        for label, score in results.items():
            print(f"{label}: {score:.4f}")
        print("\n")