import streamlit as st
import joblib
import sys
import os

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODEL_DIR = os.path.join(BASE_DIR, "models")

sys.path.append(SRC_DIR)

from preprocess import preprocess_text


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Support AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    os.path.join(MODEL_DIR, "intent_model.pkl")
)

vectorizer = joblib.load(
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)


# ============================================================
# RESPONSE DATABASE
# ============================================================

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


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        padding: 25px 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.20);
    }

    .main-header h1 {
        margin: 0;
        font-size: 32px;
        font-weight: 700;
    }

    .main-header p {
        margin: 7px 0 0 0;
        opacity: 0.90;
        font-size: 15px;
    }

    /* Chat cards */
    .user-message {
        background: #4f46e5;
        color: white;
        padding: 13px 17px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0 10px auto;
        max-width: 75%;
        width: fit-content;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15);
    }

    .bot-message {
        background: white;
        color: #1f2937;
        padding: 13px 17px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px auto 10px 0;
        max-width: 75%;
        width: fit-content;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.07);
        border: 1px solid #e5e7eb;
    }

    /* Intent card */
    .intent-card {
        background: white;
        border-left: 5px solid #4f46e5;
        padding: 14px 18px;
        border-radius: 12px;
        margin: 8px 0 18px 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }

    .intent-title {
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .intent-value {
        font-size: 17px;
        color: #312e81;
        font-weight: 700;
        margin-top: 4px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Suggested question buttons */
    .suggestion-title {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 8px;
    }

    /* Footer */
    .project-footer {
        text-align: center;
        color: #6b7280;
        font-size: 12px;
        padding: 25px 0 10px 0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🤖 Customer Support AI")

    st.markdown("---")

    st.markdown("### 📌 About")

    st.write(
        "An NLP-based customer support chatbot "
        "that classifies user queries into predefined "
        "customer service intents."
    )

    st.markdown("### 🧠 NLP Pipeline")

    st.markdown("""
    **1.** Text Preprocessing  
    **2.** TF-IDF Vectorization  
    **3.** Multinomial Naive Bayes  
    **4.** Intent Classification  
    **5.** Response Generation
    """)

    st.markdown("---")

    st.markdown("### 📊 Project Stats")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Intents", "27")

    with col2:
        st.metric("Accuracy", "99.14%")

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style="margin-top:30px; font-size:12px; opacity:0.7;">
    NLP Mini Project<br>
    Customer Support Intent Classification
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>🤖 Customer Support Assistant</h1>
    <p>AI-powered customer service using Natural Language Processing</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if len(st.session_state.messages) == 0:

    st.markdown("""
    <div style="
        background:white;
        padding:25px;
        border-radius:18px;
        text-align:center;
        border:1px solid #e5e7eb;
        box-shadow:0 4px 15px rgba(0,0,0,0.05);
        margin-bottom:20px;
    ">
        <div style="font-size:42px;">👋</div>
        <h2 style="color:#1f2937;">Hello! How can I help you?</h2>
        <p style="color:#6b7280;">
        Ask me about orders, refunds, payments, accounts,
        delivery, invoices and more.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="suggestion-title">💡 Try asking:</div>',
        unsafe_allow_html=True
    )

    suggestions = [
        "Where is my order?",
        "I forgot my password",
        "How can I get a refund?",
        "I want to cancel my order"
    ]

    cols = st.columns(4)

    for i, suggestion in enumerate(suggestions):

        with cols[i]:

            if st.button(
                suggestion,
                use_container_width=True,
                key=f"suggestion_{i}"
            ):
                st.session_state.messages.append({
                    "role": "user",
                    "content": suggestion
                })

                clean_message = preprocess_text(suggestion)

                message_vector = vectorizer.transform(
                    [clean_message]
                )

                prediction = model.predict(
                    message_vector
                )[0]

                probabilities = model.predict_proba(
                    message_vector
                )[0]

                confidence = max(probabilities) * 100

                bot_response = responses.get(
                    prediction,
                    "Sorry, I could not understand your request."
                )

                st.session_state.messages.append({
                    "role": "bot",
                    "content": bot_response,
                    "intent": prediction,
                    "confidence": confidence
                })

                st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end;">
                <div class="user-message">
                    👤 &nbsp; {message["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-start;">
                <div class="bot-message">
                    🤖 &nbsp; {message["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
    f"""
    <div class="intent-card">
        <strong>🎯 Detected Intent</strong><br><br>
        <span style="font-size:18px; color:#312e81;">
            {message["intent"]}
        </span>
        <br><br>
        <span style="color:#6b7280;">
            Classification confidence:
            <strong>{message["confidence"]:.2f}%</strong>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHAT INPUT
# ============================================================

user_message = st.chat_input(
    "Type your customer support question..."
)


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

if user_message:

    st.session_state.messages.append({
        "role": "user",
        "content": user_message
    })

    clean_message = preprocess_text(user_message)

    message_vector = vectorizer.transform(
        [clean_message]
    )

    predicted_intent = model.predict(
        message_vector
    )[0]

    probabilities = model.predict_proba(
        message_vector
    )[0]

    confidence = max(probabilities) * 100

    bot_response = responses.get(
        predicted_intent,
        "Sorry, I could not understand your request."
    )

    st.session_state.messages.append({
        "role": "bot",
        "content": bot_response,
        "intent": predicted_intent,
        "confidence": confidence
    })

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="project-footer">
    Built with Python • NLP • TF-IDF • Multinomial Naive Bayes • Streamlit
</div>
""", unsafe_allow_html=True)