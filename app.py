"""Interfata web a chatbotului.

Rulare:  python -m streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

from src.data_loader import incarca_faq, incarca_intrebari_test
from src.cautare_cuvinte_cheie import CautareCuvinteCheie
from src.cautare_hibrida import este_relevant

# Sub acest prag consideram ca nu am gasit un raspuns potrivit. E mai bine ca
# sistemul sa spuna "nu stiu" decat sa raspunda orice.
#
# Pragul e diferit pentru fiecare metoda, pentru ca scorurile lor sunt pe scale
# diferite. Valorile au fost alese testand pe intrebarile din tests/.
PRAGURI_INCREDERE = {
    "cuvinte_cheie": 0.15,
    "semantic": 0.35,
}

st.set_page_config(page_title="Chatbot FAQ", page_icon="?", layout="centered")


@st.cache_resource
def incarca_sistem_cuvinte_cheie():
    return CautareCuvinteCheie(incarca_faq())


@st.cache_resource
def incarca_sistem_semantic():
    from src.cautare_semantica import CautareSemantica
    return CautareSemantica(incarca_faq())


@st.cache_resource
def incarca_sistem_hibrid():
    from src.cautare_hibrida import CautareHibrida
    return CautareHibrida(incarca_sistem_cuvinte_cheie(), incarca_sistem_semantic())


st.title("Chatbot intrebari frecvente")
st.caption("Sistem de regasire a raspunsurilor dintr-o baza de intrebari frecvente")

with st.sidebar:
    st.header("Configurare")
    metoda = st.radio(
        "Metoda de cautare",
        ["Hibrid", "Semantica", "Cuvinte-cheie", "Comparatie"],
        help="Hibrid combina ambele metode si da cele mai bune rezultate",
    )
    numar_rezultate = st.slider("Numar de rezultate", 1, 5, 3)
    arata_scoruri = st.checkbox("Afiseaza scorurile de similaritate", value=True)

    st.divider()
    faq = incarca_faq()
    st.metric("Intrebari in baza de date", len(faq))
    categorii = sorted({item["categorie"] for item in faq})
    st.caption("Categorii: " + ", ".join(categorii))


def afiseaza_rezultate(rezultate, arata_scoruri, metoda="semantic"):
    """Afiseaza raspunsul principal si, ascunse, celelalte rezultate."""
    principal = rezultate[0]

    if metoda == "hibrid":
        relevant = este_relevant(rezultate, PRAGURI_INCREDERE["semantic"])
    else:
        relevant = principal["scor"] >= PRAGURI_INCREDERE[metoda]

    if not relevant:
        st.warning(
            "Nu am gasit un raspuns suficient de relevant pentru aceasta "
            "intrebare. Incercati o reformulare sau contactati secretariatul."
        )
        st.caption(f"Cea mai apropiata intrebare gasita: {principal['intrebare']}")
        return

    st.success(principal["raspuns"])

    detaliu = f"Intrebare din baza: {principal['intrebare']}"
    if arata_scoruri:
        detaliu += f"  |  scor: {principal['scor']:.3f}"
    st.caption(detaliu)

    if len(rezultate) > 1:
        with st.expander(f"Alte {len(rezultate) - 1} rezultate potrivite"):
            for r in rezultate[1:]:
                eticheta = r["intrebare"]
                if arata_scoruri:
                    eticheta += f"  (scor: {r['scor']:.3f})"
                st.markdown(f"**{eticheta}**")
                st.write(r["raspuns"])
                st.divider()


intrebare = st.text_input(
    "Intrebarea dumneavoastra",
    placeholder="ex: ce medie imi trebuie pentru bursa?",
)

if intrebare:
    if metoda == "Comparatie":
        coloana_stanga, coloana_dreapta = st.columns(2)

        with coloana_stanga:
            st.subheader("Cuvinte-cheie")
            sistem = incarca_sistem_cuvinte_cheie()
            afiseaza_rezultate(sistem.cauta(intrebare, k=numar_rezultate),
                               arata_scoruri, "cuvinte_cheie")

        with coloana_dreapta:
            st.subheader("Semantica")
            try:
                sistem = incarca_sistem_semantic()
                afiseaza_rezultate(sistem.cauta(intrebare, k=numar_rezultate),
                                   arata_scoruri, "semantic")
            except ImportError:
                st.error("Instalati sentence-transformers pentru cautarea semantica.")

    elif metoda == "Cuvinte-cheie":
        sistem = incarca_sistem_cuvinte_cheie()
        afiseaza_rezultate(sistem.cauta(intrebare, k=numar_rezultate),
                           arata_scoruri, "cuvinte_cheie")

    elif metoda == "Semantica":
        try:
            sistem = incarca_sistem_semantic()
            afiseaza_rezultate(sistem.cauta(intrebare, k=numar_rezultate),
                               arata_scoruri, "semantic")
        except ImportError:
            st.error("Pachetul sentence-transformers nu este instalat. "
                     "Rulati: pip install sentence-transformers")

    else:
        try:
            sistem = incarca_sistem_hibrid()
            rezultate = sistem.cauta(intrebare, k=numar_rezultate)
            afiseaza_rezultate(rezultate, arata_scoruri, "hibrid")

            if arata_scoruri:
                with st.expander("Cum a fost obtinut acest clasament"):
                    for r in rezultate:
                        st.caption(
                            f"{r['intrebare']} - pozitie cuvinte-cheie: "
                            f"{r['rang_cuvinte_cheie']}, pozitie semantica: "
                            f"{r['rang_semantic']}"
                        )
        except ImportError:
            st.error("Pachetul sentence-transformers nu este instalat. "
                     "Rulati: pip install sentence-transformers")

else:
    st.info("Scrieti o intrebare pentru a primi un raspuns.")
    st.markdown("**Exemple de intrebari:**")
    for exemplu in [
        "ce medie imi trebuie ca sa iau bursa?",
        "am picat un examen, ce se intampla?",
        "cat costa camera la camin?",
        "nu imi mai stiu parola de la mail",
    ]:
        st.markdown(f"- {exemplu}")


# ---------------------------------------------------------------------------
# Sectiunea cu rezultatele evaluarii.
#
# Fisierele sunt generate de ruleaza_evaluare.py. Le afisam si in interfata,
# ca cineva care deschide aplicatia sa poata vedea cat de bine functioneaza
# sistemul, nu doar sa il incerce.
# ---------------------------------------------------------------------------

DIRECTOR_REZULTATE = Path(__file__).resolve().parent / "rezultate"

ETICHETE = {
    "Cuvinte-cheie (TF-IDF)": "cuvinte_cheie",
    "Semantic (embeddings)": "semantic",
    "Hibrid (RRF)": "hibrid",
}


def citeste_json(nume):
    """Citeste un fisier din rezultate/, sau None daca nu exista."""
    cale = DIRECTOR_REZULTATE / nume
    if not cale.exists():
        return None
    try:
        return json.loads(cale.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


st.divider()

with st.expander("Rezultatele evaluarii pe setul de test"):
    metrici = citeste_json("metrici.json")

    if metrici is None:
        st.info(
            "Rezultatele nu au fost generate inca. "
            "Rulati `python ruleaza_evaluare.py` in folderul proiectului."
        )
    else:
        numar_test = len(incarca_intrebari_test())
        st.caption(
            f"Masurat pe {numar_test} de intrebari noi, formulate diferit "
            f"fata de cele din baza de date."
        )

        st.dataframe(
            [
                {"Metoda": nume, **{m: round(v, 3) for m, v in valori.items()}}
                for nume, valori in metrici.items()
            ],
            hide_index=True,
            width="stretch",
        )

        st.caption(
            "recall@K = de cate ori raspunsul corect apare in primele K rezultate. "
            "MRR = media lui 1/pozitie."
        )

        # Cate intrebari rezolva fiecare metoda corect din prima
        rezumat = []
        for nume in metrici:
            detalii = citeste_json(f"detalii_{ETICHETE.get(nume, '')}.json")
            if detalii:
                rezumat.append((nume, len(detalii["corecte"]), len(detalii["gresite"])))

        if rezumat:
            st.markdown("**Corecte pe prima pozitie**")
            for nume, corecte, gresite in rezumat:
                st.markdown(f"- {nume}: {corecte} corecte, {gresite} gresite")

            # Exemple de intrebari la care sistemul greseste
            nume_ales = st.selectbox(
                "Vezi exemple de raspunsuri gresite pentru:",
                [r[0] for r in rezumat],
            )
            detalii = citeste_json(f"detalii_{ETICHETE.get(nume_ales, '')}.json")
            for d in detalii["gresite"][:5]:
                pozitie = d["pozitie"] if d["pozitie"] else "negasit"
                st.markdown(
                    f"**“{d['intrebare']}”**  \n"
                    f"a returnat: *{d['intrebare_top1']}*  \n"
                    f"raspunsul corect era pe pozitia: {pozitie}"
                )
