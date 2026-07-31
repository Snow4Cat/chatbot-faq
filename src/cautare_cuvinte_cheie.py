"""Cautare prin cuvinte-cheie, folosind TF-IDF.

Compara cuvintele din intrebare cu cele din baza de date. Nu intelege
sensul: doua formulari diferite ale aceleiasi idei nu sunt legate.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import normalizeaza, STOPWORDS_RO


class CautareCuvinteCheie:
    def __init__(self, faq):
        self.faq = faq

        # Indexam si intrebarea si raspunsul, pentru ca uneori cuvantul
        # cautat apare doar in raspuns.
        documente = [
            normalizeaza(item["intrebare"] + " " + item["raspuns"])
            for item in faq
        ]

        # ngram_range=(1, 2) prinde si expresii de doua cuvinte,
        # de exemplu "taxa scolarizare".
        self.vectorizator = TfidfVectorizer(
            stop_words=STOPWORDS_RO,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrice = self.vectorizator.fit_transform(documente)

    def cauta(self, intrebare, k=3):
        """Returneaza primele k rezultate, de la cel mai bun la cel mai slab."""
        vector = self.vectorizator.transform([normalizeaza(intrebare)])
        scoruri = cosine_similarity(vector, self.matrice)[0]

        indici = np.argsort(scoruri)[::-1][:k]
        return [
            {
                "id": self.faq[i]["id"],
                "intrebare": self.faq[i]["intrebare"],
                "raspuns": self.faq[i]["raspuns"],
                "categorie": self.faq[i]["categorie"],
                "scor": float(scoruri[i]),
            }
            for i in indici
        ]
