# AfricaAI – Chatbot éducatif multilingue (prototype)

AfricaAI est un prototype de chatbot éducatif qui répond aux questions d'élèves dans plusieurs langues, en s'appuyant sur une **recherche sémantique** dans un corpus de questions-réponses scolaires. L'objectif à terme : réduire les barrières linguistiques dans l'éducation, avec le Bénin comme pays pilote.

> **Statut : prototype.** Ce dépôt contient une première version fonctionnelle de la recherche sémantique et de l'interface. Les fonctionnalités listées dans la [feuille de route](#feuille-de-route-non-implémenté) ne sont **pas encore implémentées**.

## Ce qui fonctionne aujourd'hui

- **Recherche sémantique** : les questions du corpus sont transformées en embeddings (`sentence-transformers/all-MiniLM-L6-v2`) et indexées avec **FAISS**. La question de l'élève est comparée à l'index et les 3 résultats les plus proches sont proposés, avec un score de similarité.
- **Filtres** par langue, matière et niveau scolaire.
- **Repli sur un LLM** (`google/flan-t5-xl`) quand aucun résultat n'est assez proche, avec un avertissement à l'utilisateur.
- **Synthèse vocale** optionnelle (gTTS).
- **Détection de la langue** de la question (`langdetect`).
- **Historique de session** dans l'interface Streamlit.
- **Pipeline d'indexation reproductible** : `build_index.py` reconstruit l'index et les métadonnées à partir du corpus JSON.

## Corpus

Le corpus (`chatbot_corpus_multilingue_niveaux.json`) est une **base de démonstration de 41 questions-réponses** :

| Langue | Code | Questions |
|---|---|---|
| Français | `fr` | 16 |
| Anglais | `en` | 15 |
| Swahili | `sw` | 4 |
| Haoussa | `ha` | 3 |
| Yoruba | `yo` | 3 |

Matières : maths, sciences, histoire. Niveaux : de CM2 à la 3e (les langues locales ne couvrent pour l'instant que quelques questions de CM2 et de 6e).

## Limites connues

- Le modèle d'embeddings `all-MiniLM-L6-v2` est principalement entraîné sur l'anglais : la recherche est **moins fiable en yoruba, haoussa et swahili**.
- Le corpus est petit, et le seuil de similarité n'est pas calibré : certaines questions proches peuvent renvoyer une réponse hors sujet.
- Le LLM de repli (`flan-t5-xl`) est lourd (plusieurs Go de mémoire) et ses réponses **ne sont pas vérifiées**.
- La synthèse vocale gTTS ne couvre pas forcément toutes les langues du corpus.
- L'onglet « Tableau de bord » est un espace réservé.

## Architecture actuelle

```
Question ──► embeddings (MiniLM) ──► recherche FAISS ──► filtres (langue/matière/niveau)
                                          │
                                          ├─ similarité suffisante ─► réponse du corpus (+ TTS optionnel)
                                          └─ similarité faible ─────► repli LLM (flan-t5-xl)
```

Interface : Streamlit.

## Installation

```bash
git clone https://github.com/Emperor-Lobo/AfricaAI.git
cd AfricaAI
python -m venv .venv && source .venv/bin/activate

pip install streamlit sentence-transformers faiss-cpu transformers sentencepiece \
            torch gtts langdetect pandas numpy

python build_index.py      # génère educ_index.faiss, educ_embeddings.npy, educ_metadatas.json
streamlit run app.py
```

Au premier lancement, les modèles sont téléchargés depuis Hugging Face (`flan-t5-xl` pèse plusieurs Go).

## Structure du dépôt

```
AfricaAI/
├── app.py                                  # interface Streamlit
├── build_index.py                          # construction de l'index FAISS
├── chatbot_corpus_multilingue_niveaux.json # corpus (41 Q&R)
├── educ_index.faiss                        # index (généré)
├── educ_embeddings.npy                     # embeddings (générés)
├── educ_metadatas.json                     # métadonnées (générées)
├── africa_style.css                        # style de l'interface
└── logo_africaai.svg
```

## Feuille de route (non implémenté)

- [ ] Modèle d'embeddings multilingue pour de meilleurs résultats en langues locales
- [ ] Traduction automatique (NLLB / MarianMT)
- [ ] Reconnaissance vocale (Whisper)
- [ ] Agrandir le corpus et le faire valider par des locuteurs natifs
- [ ] Ajout du fon
- [ ] Vrai tableau de bord de suivi (questions sans réponse fiable, langues utilisées)
- [ ] Quiz interactifs par matière et niveau
- [ ] Mode hors-ligne (PWA)
- [ ] API (FastAPI) et base de données (PostgreSQL)
- [ ] Conteneurisation (Docker) et intégration continue (GitHub Actions)

## Auteur

Amos Clegbaza – [GitHub](https://github.com/Emperor-Lobo)
