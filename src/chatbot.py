import joblib

from preprocess import preprocess_text


# Load trained model
model = joblib.load("models/intent_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# Predefined chatbot responses
responses = {
    "cancel_order":
        "You can cancel your order if it is eligible for cancellation.",

    "change_order":
        "You can modify your order if it has not been processed yet.",

    "change_shipping_address":
        "You can change your shipping address from your order details if the order has not been shipped.",

    "check_cancellation_fee":
        "Cancellation fees depend on the order status and applicable policy.",

    "check_invoice":
        "You can check your invoice from the order details section.",

    "check_payment_methods":
        "We support several payment methods. Please check the available options during checkout.",

    "check_refund_policy":
        "Refund eligibility and processing time depend on the refund policy.",

    "complaint":
        "We are sorry for the inconvenience. Please provide more details about your complaint.",

    "contact_customer_service":
        "You can contact our customer service team for further assistance.",

    "contact_human_agent":
        "I can connect you with a human support agent for further assistance.",

    "create_account":
        "You can create a new account by selecting the registration option.",

    "delete_account":
        "You can request account deletion through your account settings.",

    "delivery_options":
        "Available delivery options are shown during the checkout process.",

    "delivery_period":
        "Delivery time depends on your location and selected shipping method.",

    "edit_account":
        "You can update your account information from your account settings.",

    "get_invoice":
        "Your invoice can be downloaded from the order details section.",

    "get_refund":
        "Refunds are processed according to the refund policy and may take some time to appear.",

    "newsletter_subscription":
        "You can manage your newsletter subscription from your account settings.",

    "payment_issue":
        "Please check your payment details and try the payment again. If the problem continues, contact support.",

    "place_order":
        "You can place an order by selecting a product and completing the checkout process.",

    "recover_password":
        "You can reset your password using the 'Forgot Password' option.",

    "registration_problems":
        "Please check your registration details and try again. If the problem continues, contact support.",

    "review":
        "You can leave a review after purchasing the product.",

    "set_up_shipping_address":
        "You can add a shipping address from your account or during checkout.",

    "switch_account":
        "You can switch accounts by logging out and signing in with another account.",

    "track_order":
        "You can track your order using your order details or tracking information.",

    "track_refund":
        "You can check your refund status using your refund or order details."
}


# Function to predict intent
def predict_intent(message):

    # Preprocess message
    clean_message = preprocess_text(message)

    # Convert message into TF-IDF
    message_vector = vectorizer.transform([clean_message])

    # Predict intent
    predicted_intent = model.predict(message_vector)[0]

    return predicted_intent


# Start chatbot
print("======================================")
print("     CUSTOMER SUPPORT CHATBOT")
print("======================================")
print("Type 'exit' to stop the chatbot.\n")


while True:

    user_message = input("You: ")

    if user_message.lower() == "exit":
        print("Bot: Thank you for using our customer support chatbot!")
        break

    intent = predict_intent(user_message)

    response = responses.get(
        intent,
        "Sorry, I could not understand your request."
    )

    print("Detected Intent:", intent)
    print("Bot:", response)
    print()