import pickle
from io import BytesIO  # For in-memory file handling

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud


# ------------------- Load Model & Tokenizer -------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("../models/cnn_model.h5")
    with open("../models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


# ------------------- Predict Function -------------------
def predict_sentiment(texts, tokenizer, model, max_length=400):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences, maxlen=max_length, padding="post", truncating="post"
    )
    predictions = model.predict(padded)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels


# ------------------- Word Cloud -------------------
def generate_wordcloud(texts):
    text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


# ------------------- Excel Export -------------------
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sentiment Results")
    output.seek(0)
    return output


# ------------------- App Layout -------------------
def main():
    st.set_page_config(page_title="Customer Review Analyzer", layout="wide")
    st.title("📊 Customer Review Sentiment Analysis Dashboard")
    st.markdown(
        "This app helps businesses understand customer feedback using a CNN sentiment model."
    )

    model, tokenizer = load_model_and_tokenizer()
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    st.sidebar.header("🔽 Upload Excel File")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a file with a 'review' column", type=["xlsx"]
    )

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        if "review" not in df.columns:
            st.error("❌ Excel file must contain a 'review' column.")
            return

        df["review"] = df["review"].astype(str)
        df["Sentiment"] = predict_sentiment(df["review"], tokenizer, model)
        df["Sentiment_Label"] = df["Sentiment"].map(label_map)

        st.success("✅ Sentiment Prediction Complete!")

        with st.container():
            st.subheader("📋 Sample Reviews with Predictions")
            st.dataframe(
                df[["review", "Sentiment_Label"]].head(10), use_container_width=True
            )

        # ------------------- Download Button -------------------
        st.markdown("### 📥 Download Results")
        excel_data = convert_df_to_excel(df[["review", "Sentiment_Label"]])
        st.download_button(
            label="⬇️ Download Excel Report",
            data=excel_data,
            file_name="sentiment_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # ------------------- Graph Layout -------------------
        st.subheader("📈 Sentiment Analysis Summary")
        col1, col2 = st.columns(2)

        sentiment_counts = df["Sentiment_Label"].value_counts()

        with col1:
            st.markdown("#### Sentiment Distribution (Bar Chart)")
            st.bar_chart(sentiment_counts)

        with col2:
            st.markdown("#### Sentiment Distribution (Pie Chart)")
            fig1, ax1 = plt.subplots()
            ax1.pie(
                sentiment_counts,
                labels=sentiment_counts.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=["#ff6b6b", "#feca57", "#1dd1a1"],
            )
            ax1.axis("equal")
            st.pyplot(fig1)

        # ------------------- Word Cloud -------------------
        st.subheader("☁️ Frequently Used Words (Word Cloud)")
        generate_wordcloud(df["review"])

    # ------------------- Single Review Input -------------------
    st.markdown("---")
    st.subheader("🔍 Analyze a Single Review")
    user_input = st.text_area("Enter a customer review:")
    if st.button("Analyze"):
        if user_input.strip():
            pred = predict_sentiment([user_input], tokenizer, model)
            label = label_map[pred[0]]
            st.success(f"🧠 Predicted Sentiment: **{label}**")
        else:
            st.warning("Please enter a review before analyzing.")


if __name__ == "__main__":
    main()
