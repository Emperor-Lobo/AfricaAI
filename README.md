#  AfricaAI – Chatbot Éducatif Multilingue pour l’Afrique

##  Description
AfricaAI est une plateforme éducative basée sur l’Intelligence Artificielle qui vise à offrir un **accès équitable, multilingue et intelligent à une éducation de qualité** pour les élèves africains, en particulier au Bénin comme pays pilote.

Le projet combine :
- **LLM open source** (FLAN-T5, LLaMA, etc.)
- **Recherche sémantique** avec FAISS
- **Traduction automatique** (MarianMT, NLLB)
- **Synthèse vocale (TTS)** pour accessibilité
- **Interface simple et responsive** (Streamlit)

---

##  Fonctionnalités
- Chatbot éducatif multilingue (français, anglais, yoruba, fon, haoussa, swahili).
- Corpus de plus de **5 000 questions-réponses** alignées aux programmes scolaires.
- **Quiz interactifs** par matière et niveau.
- Mode **hors-ligne** via PWA et packs préchargés.
- Tableau de bord pour suivi pédagogique.
- Accessibilité via **voix (TTS)** et support des langues locales.

---

##  Architecture
- **Frontend** : Streamlit (PWA responsive)
- **Backend** : FastAPI + orchestrateur IA
- **NLP/LLM** : FLAN-T5 / LLaMA / Sentence Transformers
- **Index sémantique** : FAISS
- **Traduction** : MarianMT, NLLB
- **TTS** : gTTS / Coqui-TTS
- **Base de données** : PostgreSQL + JSON corpus

---

##  Structure du projet
```bash
AfricaAI/
├── app.py
├── build_index.py
├── chatbot_corpus_multilingue_niveaux.json
├── educ_embeddings.npy
├── educ_index.faiss
├── educ_metadatas.json
├── africa_style.css
├── logo_africaai.svg
├── logs_chatbot.csv
├── requirements.txt

