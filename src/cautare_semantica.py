"""Cautare semantica, pe baza de embeddings.

Fiecare intrebare devine un vector care codifica sensul propozitiei.
Doua formulari diferite ale aceleiasi idei dau vectori apropiati, chiar
daca nu au niciun cuvant comun.
"""

import pickle
from pathlib import Path

import numpy as np

# Modelul trebuie sa fie multilingv, pentru ca datele sunt in romana.
# Modelele doar pentru engleza dau rezultate slabe aici.
MODEL_IMPLICIT = "paraphrase-multilingual-MiniLM-L12-v2"

CALE_CACHE = Path(__file__).resolve().parent.parent / "data" / "embeddings.pkl"


class CautareSemantica:
    def __init__(self, faq, nume_model=MODEL_IMPLICIT, foloseste_cache=True,
                 foloseste_faiss=False):
        from sentence_transformers import SentenceTransformer

        self.faq = faq
        self.nume_model = nume_model
        self.model = SentenceTransformer(nume_model)

        documente = [item["intrebare"] + " " + item["raspuns"] for item in faq]
        self.embeddings = self._obtine_embeddings(documente, foloseste_cache)

        self.index = None
        if foloseste_faiss:
            self._construieste_index_faiss()

    def _obtine_embeddings(self, documente, foloseste_cache):
        """Calculeaza vectorii o singura data si ii salveaza pentru rularile urmatoare."""
        cheie = (self.nume_model, len(documente), hash(tuple(documente)))

        if foloseste_cache and CALE_CACHE.exists():
            with open(CALE_CACHE, "rb") as f:
                salvat = pickle.load(f)
            if salvat.get("cheie") == cheie:
                return salvat["embeddings"]

        # normalize_embeddings=True aduce vectorii la lungime 1, ca sa putem
        # calcula similaritatea cosinus printr-o simpla inmultire.
        embeddings = self.model.encode(
            documente,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        if foloseste_cache:
            CALE_CACHE.parent.mkdir(exist_ok=True)
            with open(CALE_CACHE, "wb") as f:
                pickle.dump({"cheie": cheie, "embeddings": embeddings}, f)

        return embeddings

    def _construieste_index_faiss(self):
        """Indexare optionala cu FAISS.

        Pe 40 de intrebari nu aduce niciun castig. Este inclusa ca sa arate
        cum s-ar face cautarea pe seturi de date mari.
        """
        import faiss

        dimensiune = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimensiune)
        self.index.add(np.asarray(self.embeddings, dtype="float32"))

    def cauta(self, intrebare, k=3):
        """Returneaza primele k rezultate, de la cel mai bun la cel mai slab."""
        vector = self.model.encode([intrebare], normalize_embeddings=True)

        if self.index is not None:
            scoruri, indici = self.index.search(
                np.asarray(vector, dtype="float32"), k
            )
            scoruri, indici = scoruri[0], indici[0]
        else:
            # Vectorii au lungime 1, deci inmultirea da direct similaritatea cosinus.
            toate = self.embeddings @ vector[0]
            indici = np.argsort(toate)[::-1][:k]
            scoruri = toate[indici]

        return [
            {
                "id": self.faq[i]["id"],
                "intrebare": self.faq[i]["intrebare"],
                "raspuns": self.faq[i]["raspuns"],
                "categorie": self.faq[i]["categorie"],
                "scor": float(s),
            }
            for i, s in zip(indici, scoruri)
        ]
