
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Chargement du JSON
with open("chatbot_corpus_multilingue_niveaux.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Modèle d'embedding
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

texts = []
metadatas = []

# Parcours de toutes les langues, matières et niveaux
for lang, subjects in data.items():
    for subject, levels in subjects.items():
        for level, qas in levels.items():
            for question_key, answer in qas.items():
                question_text = question_key.replace("_", " ")
                texts.append(question_text)
                metadatas.append({
                    "lang": lang,
                    "subject": subject,
                    "level": level,
                    "question": question_text,
                    "answer": answer
                })

# Génération des embeddings
embeddings = model.encode(texts, convert_to_numpy=True)

# Construction de l'index FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Sauvegarde
np.save("educ_embeddings.npy", embeddings)
with open("educ_metadatas.json", "w", encoding="utf-8") as f:
    json.dump(metadatas, f, ensure_ascii=False, indent=2)
faiss.write_index(index, "educ_index.faiss")

print("Index et métadonnées sauvegardés.")
