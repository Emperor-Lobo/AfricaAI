import streamlit as st
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
from gtts import gTTS
from io import BytesIO
from transformers import pipeline
from langdetect import detect
import pandas as pd
from datetime import datetime

# 🌿 Charger le CSS
with open("africa_style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 🌍 Sidebar avec logo et description
st.sidebar.image("logo_africaai.svg", width=120)
st.sidebar.markdown("🌿 **AfricaAI - Éducation pour Tous**")

# Traductions de l'interface
labels = {
    "fr": {
        "title": " Chatbot Éducatif Multilingue",
        "description": "Pose une question en choisissant la langue, la matière et le niveau scolaire.",
        "lang": "Langue :",
        "subject": "Matière :",
        "level": "Niveau scolaire :",
        "question": "Ta question :",
        "results": "Résultats les plus proches :",
        "history": "📝 Historique de la session",
        "interface_lang": "Langue de l'interface :",
        "dashboard_tab": "📊 Tableau de bord"
    }
}

interface_lang = "fr"
L = labels[interface_lang]

# 🔹 Chargement des modèles
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_data
def load_index():
    index = faiss.read_index("educ_index.faiss")
    with open("educ_metadatas.json", "r", encoding="utf-8") as f:
        metadatas = json.load(f)
    return index, metadatas

@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-xl"
    )

model = load_model()
index, metadatas = load_index()
llm_generator = load_llm()

# 🔹 Interface principale
tab1, tab2 = st.tabs([L["title"], L["dashboard_tab"]])

# 🟢 Onglet Chatbot
with tab1:
    st.title(L["title"])
    st.markdown(L["description"])

    langs = sorted(set(m["lang"] for m in metadatas))
    subjects = sorted(set(m["subject"] for m in metadatas))
    levels = sorted(set(m["level"] for m in metadatas))

    col1, col2, col3 = st.columns(3)
    with col1:
        lang_filter = st.selectbox(L["lang"], ["Toutes"] + langs)
    with col2:
        subject_filter = st.selectbox(L["subject"], ["Toutes"] + subjects)
    with col3:
        level_filter = st.selectbox(L["level"], ["Tous"] + levels)

    auto_tts = st.checkbox("🔈 Activer la lecture automatique")

    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input(L["question"])

    if question.strip():
        detected_lang = detect(question)
        st.markdown(f"🌐 Langue détectée : `{detected_lang}`")

        q_embedding = model.encode([question])
        D, I = index.search(np.array(q_embedding), k=3)

        st.markdown(f"### {L['results']}")
        found_any = False
        best_score = 0

        for rank, idx in enumerate(I[0]):
            m = metadatas[idx]
            distance = D[0][rank]
            score = 1 - distance / 2

            if rank == 0:
                best_score = score

            if score < 0.5:
                continue

            if (lang_filter != "Toutes" and m["lang"] != lang_filter):
                continue
            if (subject_filter != "Toutes" and m["subject"] != subject_filter):
                continue
            if (level_filter != "Tous" and m["level"] != level_filter):
                continue

            found_any = True

            if rank == 0:
                st.success(
                    f"**Réponse principale (similarité {score:.2f}) :**\n\n"
                    f"**Langue :** {m['lang']}  \n"
                    f"**Matière :** {m['subject']}  \n"
                    f"**Niveau :** {m['level']}  \n"
                    f"**Question proche :** {m['question']}  \n\n"
                    f"**Réponse :** {m['answer']}"
                )

                if auto_tts:
                    tts_q = gTTS(question, lang=detected_lang)
                    buf_q = BytesIO()
                    tts_q.write_to_fp(buf_q)
                    st.audio(buf_q.getvalue(), format="audio/mp3")

                    tts = gTTS(m["answer"], lang=m["lang"])
                    buf = BytesIO()
                    tts.write_to_fp(buf)
                    st.audio(buf.getvalue(), format="audio/mp3")

                st.session_state.history.append((question, m["answer"]))

            else:
                with st.expander(f"Suggestion {rank+1} (similarité {score:.2f})"):
                    st.markdown(f"- Langue : {m['lang']}")
                    st.markdown(f"- Matière : {m['subject']}")
                    st.markdown(f"- Niveau : {m['level']}")
                    st.markdown(f"- Question proche : {m['question']}")
                    st.markdown(f"**Réponse :** {m['answer']}")

        # 🎯 Fallback LLM avec prompt amélioré
        if not found_any or best_score < 0.75:
            st.warning("🤔 La réponse trouvée est imprécise ou inexistante. Je vais générer une réponse avec le LLM...")
            with st.spinner("Je réfléchis..."):
                prompt = (
                    f"Réponds à la question suivante de manière simple et claire, sans répéter la question.\n\n"
                    f"Question: {question}\n"
                    "Réponse:"
                )
                result = llm_generator(
                    prompt,
                    max_length=80,
                    do_sample=False,
                    clean_up_tokenization_spaces=True
                )
                llm_answer = result[0]["generated_text"].strip()

            st.success(llm_answer)

            if auto_tts:
                tts = gTTS(llm_answer, lang="fr")
                buf = BytesIO()
                tts.write_to_fp(buf)
                st.audio(buf.getvalue(), format="audio/mp3")

            st.session_state.history.append((question, llm_answer))

    st.subheader(L["history"])
    if st.session_state.history:
        if st.button("🗑️ Vider l'historique"):
            st.session_state.history = []
        for q, r in st.session_state.history[-10:]:
            st.markdown(f"**Tu :** {q}")
            st.markdown(f"**Bot :** {r}")
    else:
        st.write("Aucune interaction encore.")

# 🟢 Onglet Dashboard
with tab2:
    st.title(L["dashboard_tab"])
    st.info("📊 Tableau de bord en cours de développement.")
