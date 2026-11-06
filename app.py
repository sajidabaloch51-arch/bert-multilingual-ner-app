
import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

# Updated to your verified Hugging Face repository
MODEL_NAME = "sajida12/bert-english-ner"

st.set_page_config(
    page_title="BERT English NER",
    page_icon="🔍",
    layout="centered"
)

st.title("English Named Entity Recognition")
st.write(
    "Enter an English sentence to identify people, "
    "organizations and locations."
)

@st.cache_resource
def load_ner_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_ner_model()

sentence = st.text_area(
    "Enter your sentence:",
    "A student studies at the University of Gwadar in Pakistan."
)

if st.button("Identify Entities"):

    if sentence.strip():

        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(
            outputs.logits,
            dim=2
        )[0].tolist()

        words = sentence.split()

        tokenized_words = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=128
        )

        word_ids = tokenized_words.word_ids()

        entities = []
        previous_word_id = None

        for prediction, word_id in zip(
            predictions, word_ids
        ):

            if word_id is None:
                continue

            if word_id == previous_word_id:
                continue

            label = model.config.id2label[prediction]

            if label != "O":
                entities.append({
                    "Word": words[word_id],
                    "Entity": label
                })

            previous_word_id = word_id

        if entities:
            st.subheader("Detected entities")
            st.table(entities)
        else:
            st.info("No named entities were detected.")

    else:
        st.warning("Please enter a sentence.")
