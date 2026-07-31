"""Metoda hibrida: combina cele doua cautari.

Cele doua metode gresesc pe intrebari diferite, deci se completeaza.
Combinarea se face dupa pozitii, nu dupa scoruri, pentru ca scorurile
sunt pe scale diferite (0.11-0.28 la TF-IDF, 0.19-0.78 la embeddings).

Formula (Reciprocal Rank Fusion):
    scor(document) = suma pe metode de  greutate / (k + pozitie)
"""


class CautareHibrida:
    def __init__(self, sistem_cuvinte_cheie, sistem_semantic, k=60,
                 greutate_cuvinte_cheie=1.0, greutate_semantic=1.5):
        self.sistem_cuvinte_cheie = sistem_cuvinte_cheie
        self.sistem_semantic = sistem_semantic
        self.k = k
        self.greutate_cuvinte_cheie = greutate_cuvinte_cheie
        self.greutate_semantic = greutate_semantic

    def cauta(self, intrebare, k=3):
        """Returneaza primele k rezultate dupa combinarea celor doua clasamente."""
        # Luam mai multe candidate decat returnam, ca sa aiba din ce alege.
        adancime = max(10, k * 3)

        rezultate_kw = self.sistem_cuvinte_cheie.cauta(intrebare, k=adancime)
        rezultate_sem = self.sistem_semantic.cauta(intrebare, k=adancime)

        scoruri = {}
        detalii = {}

        for rezultate, greutate in [
            (rezultate_kw, self.greutate_cuvinte_cheie),
            (rezultate_sem, self.greutate_semantic),
        ]:
            for rang, r in enumerate(rezultate, start=1):
                scoruri[r["id"]] = scoruri.get(r["id"], 0.0) + greutate / (self.k + rang)
                detalii.setdefault(r["id"], r)

        # Retinem si scorurile initiale, necesare pentru decizia de respingere.
        scor_maxim_kw = rezultate_kw[0]["scor"] if rezultate_kw else 0.0
        scor_maxim_sem = rezultate_sem[0]["scor"] if rezultate_sem else 0.0

        ordonate = sorted(scoruri.items(), key=lambda pereche: -pereche[1])[:k]

        rezultat_final = []
        for doc_id, scor in ordonate:
            item = dict(detalii[doc_id])
            item["scor"] = scor
            # Pozitiile din fiecare metoda, folosite in interfata.
            item["rang_cuvinte_cheie"] = _pozitie(rezultate_kw, doc_id)
            item["rang_semantic"] = _pozitie(rezultate_sem, doc_id)
            item["scor_maxim_cuvinte_cheie"] = scor_maxim_kw
            item["scor_maxim_semantic"] = scor_maxim_sem
            rezultat_final.append(item)

        return rezultat_final


def este_relevant(rezultate, prag_semantic):
    """Decide daca raspunsul se afiseaza sau nu.

    Ne uitam la scorul metodei semantice, nu la cel al fuziunii. Scorul
    fuziunii arata doar cat de mult sunt de acord cele doua clasamente,
    nu cat de potrivit e raspunsul, asa ca poate fi mare si la intrebari
    care nu au raspuns in baza de date.
    """
    if not rezultate:
        return False

    return rezultate[0]["scor_maxim_semantic"] >= prag_semantic


def _pozitie(rezultate, doc_id):
    """Pe ce pozitie apare documentul intr-un clasament, sau None."""
    for rang, r in enumerate(rezultate, start=1):
        if r["id"] == doc_id:
            return rang
    return None
