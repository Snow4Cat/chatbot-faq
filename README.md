# Chatbot Q&A pentru intrebari frecvente

Sistem care raspunde automat la intrebarile studentilor, pe baza unei colectii
de intrebari si raspunsuri frecvente ale unei facultati.

## Descrierea problemei

Secretariatele primesc mereu aceleasi intrebari, dar formulate altfel de fiecare
data. O cautare obisnuita prin cuvinte-cheie nu functioneaza in aceste cazuri:
intrebarea "ce medie imi trebuie ca sa iau bani lunar de la facultate" nu are
niciun cuvant comun cu "Care este media minima pentru bursa de merit?", desi
inseamna acelasi lucru.

Proiectul compara doua metode de cautare si o combinatie a lor:

1. **Cuvinte-cheie** (varianta de baza) - TF-IDF si similaritate cosinus.
   Compara cuvintele, fara sa inteleaga sensul.
2. **Semantica** - fiecare intrebare devine un vector care codifica sensul
   propozitiei, deci formulari diferite ale aceleiasi idei se potrivesc.
3. **Hibrid** - combina cele doua clasamente. A fost adaugat dupa ce am
   observat ca primele doua gresesc pe intrebari diferite.

## Tehnologii utilizate

| Componenta | Tehnologie |
|---|---|
| Varianta de baza | scikit-learn (TfidfVectorizer, cosine_similarity) |
| Embeddings | sentence-transformers, `paraphrase-multilingual-MiniLM-L12-v2` |
| Metoda hibrida | Reciprocal Rank Fusion |
| Indexare optionala | FAISS |
| Interfata | Streamlit |
| Limbaj | Python 3.9+ |

Modelul de embeddings este **multilingv**, pentru ca datele sunt in romana.
Modelele antrenate doar pe engleza dau rezultate mult mai slabe.

Proiectul nu foloseste servicii cu plata. Totul ruleaza local.

## Structura proiectului

```
chatbot-faq/
├── README.md
├── requirements.txt
├── app.py                        # interfata Streamlit
├── ruleaza_evaluare.py           # compara metodele si scrie rezultatele
├── .streamlit/config.toml        # setari Streamlit
├── data/faq.json                 # 40 de perechi intrebare-raspuns
├── src/
│   ├── data_loader.py            # incarcare date, curatare text
│   ├── cautare_cuvinte_cheie.py  # varianta de baza
│   ├── cautare_semantica.py      # embeddings
│   ├── cautare_hibrida.py        # combinarea celor doua
│   └── evaluare.py               # recall@K, MRR
├── tests/intrebari_test.json     # 25 de intrebari pentru evaluare
└── rezultate/                    # se completeaza la rulare
```

## Instalare

```bash
cd chatbot-faq
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

La prima rulare a cautarii semantice se descarca automat modelul (~120 MB),
o singura data.

Pachetul `faiss-cpu` este optional si este comentat in `requirements.txt`.

## Rulare

**Evaluarea si compararea metodelor:**

```bash
python ruleaza_evaluare.py
```

Afiseaza tabelul cu metrici si scrie in `rezultate/`:
- `metrici.txt` - tabelul comparativ
- `detalii_*.json` - exemple corecte si gresite, pentru fiecare metoda

Optiuni:

```bash
python ruleaza_evaluare.py --fara-semantic   # doar varianta de baza
python ruleaza_evaluare.py --faiss           # cu indexare FAISS
```

**Interfata web:**

```bash
python -m streamlit run app.py
```

Se deschide la `http://localhost:8501`. Din bara laterala se alege metoda;
modul **Comparatie** arata rezultatele a doua metode una langa alta.

La finalul paginii, sectiunea **Rezultatele evaluarii** afiseaza tabelul cu
metrici si exemple de raspunsuri gresite, citite din `rezultate/`. Apare doar
dupa ce ai rulat `ruleaza_evaluare.py` cel putin o data.

## Sursa datelor

Setul de date a fost construit manual pentru acest proiect si contine 40 de
perechi intrebare-raspuns, in 12 categorii: admitere, taxe, burse, examene,
cazare, secretariat, practica, licenta, orar, biblioteca, IT si mobilitate.
Continutul este realist, dar fictiv - cifrele si termenele nu corespund unei
institutii reale.

Setul de test are 25 de intrebari noi, scrise intentionat in limbaj de zi cu
zi, diferit de cel din baza de date, ca sa testeze cat de bine se descurca
sistemul cu formulari pe care nu le-a mai vazut.

## Exemplu de utilizare

```python
from src.data_loader import incarca_faq
from src.cautare_semantica import CautareSemantica

sistem = CautareSemantica(incarca_faq())
rezultate = sistem.cauta("ce medie imi trebuie pentru bursa?", k=3)

print(rezultate[0]["raspuns"])
print(f"scor: {rezultate[0]['scor']:.3f}")
```

## Principalele rezultate

Pe setul de 25 de intrebari de test:

```
Metoda                    recall@1    recall@3    recall@5         MRR
----------------------------------------------------------------------
Cuvinte-cheie (TF-IDF)       0.440       0.760       0.760       0.580
Semantic (embeddings)        0.600       0.800       0.920       0.723
Hibrid (RRF)                 0.720       0.800       0.800       0.760
```

**Cautarea semantica e mai buna pe toate metricile.** Diferenta cea mai mare e
la recall@5: varianta pe cuvinte-cheie nu gaseste deloc raspunsul la 6 din 25
de intrebari, fata de 2 la varianta semantica.

**Cele doua metode se completeaza.** Din cele 25 de intrebari, 7 sunt rezolvate
corect de ambele, 8 doar de varianta semantica, 4 doar de cea pe cuvinte-cheie
si 6 de niciuna. De aici a venit ideea metodei hibride.

**Metoda hibrida castiga la varf, dar pierde in adancime:** cel mai bun
recall@1 (0.720), dar un recall@5 mai slab decat varianta semantica (0.800 fata
de 0.920). Pentru un chatbot care arata un singur raspuns, hibridul e alegerea
buna; pentru unul care arata o lista, varianta semantica.

**Sistemul poate gresi cu incredere mare.** La intrebari care nu au raspuns in
baza de date, scorurile raman uneori ridicate. De aceea interfata are un prag
sub care refuza sa raspunda, iar pentru metoda hibrida decizia se ia dupa
scorul metodei semantice, nu dupa scorul fuziunii.

## Imbunatatiri posibile

- folosirea unui stemmer pentru romana, ca sa lege formele aceluiasi cuvant
  ("resetez" si "resetarea" sunt acum termeni diferiti);
- adaugarea de formulari alternative pentru fiecare intrare din baza;
- un set de validare separat pentru alegerea pragurilor;
- un buton prin care utilizatorul semnaleaza raspunsurile gresite.
