"""Incarcarea datelor si curatarea textului."""

import json
import re
import unicodedata
from pathlib import Path

RADACINA = Path(__file__).resolve().parent.parent
CALE_FAQ = RADACINA / "data" / "faq.json"
CALE_TEST = RADACINA / "tests" / "intrebari_test.json"


def incarca_faq(cale=CALE_FAQ):
    """Citeste intrebarile si raspunsurile din fisierul JSON."""
    with open(cale, encoding="utf-8") as f:
        date = json.load(f)

    ids = [item["id"] for item in date]
    if len(ids) != len(set(ids)):
        raise ValueError("Exista id-uri duplicate in setul de date")

    return date


def incarca_intrebari_test(cale=CALE_TEST):
    """Citeste intrebarile folosite la evaluare."""
    with open(cale, encoding="utf-8") as f:
        return json.load(f)


def elimina_diacritice(text):
    """Inlocuieste literele cu diacritice cu litere simple."""
    descompus = unicodedata.normalize("NFD", text)
    return "".join(c for c in descompus if unicodedata.category(c) != "Mn")


def normalizeaza(text):
    """Litere mici, fara diacritice si fara semne de punctuatie.

    Utilizatorii scriu des fara diacritice, asa ca uniformizam textul
    inainte de comparare.
    """
    text = elimina_diacritice(text.lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Cuvinte prea frecvente ca sa ajute la cautare. scikit-learn nu are
# o lista pentru limba romana, asa ca am scris-o manual.
STOPWORDS_RO = [
    "si", "sa", "se", "la", "de", "in", "un", "o", "cu", "pe", "ce", "care",
    "este", "sunt", "am", "ai", "are", "pot", "poti", "poate", "mai", "din",
    "pentru", "cum", "cand", "unde", "cat", "cata", "cati", "cate", "daca",
    "dar", "sau", "ca", "nu", "eu", "mi", "imi", "ma", "isi", "lui", "ei",
    "al", "ale", "ai", "a", "il", "le", "lor", "meu", "mea", "trebuie",
    "vreau", "as", "fi", "fac", "face", "acum", "aici", "asta", "acest",
]
