"""Compara cele trei metode de cautare pe setul de test.

Scrie tabelul cu metrici si exemplele corecte/gresite in directorul rezultate/.
"""

import argparse
import json
from pathlib import Path

from src.data_loader import incarca_faq, incarca_intrebari_test
from src.cautare_cuvinte_cheie import CautareCuvinteCheie
from src.evaluare import evalueaza, separa_corecte_gresite

DIRECTOR_REZULTATE = Path(__file__).resolve().parent / "rezultate"


def tabel_comparativ(metrici_pe_sistem):
    """Aranjeaza metricile intr-un tabel text."""
    nume_metrici = list(next(iter(metrici_pe_sistem.values())).keys())

    antet = f"{'Metoda':<22}" + "".join(f"{m:>12}" for m in nume_metrici)
    linii = [antet, "-" * len(antet)]

    for nume, metrici in metrici_pe_sistem.items():
        linii.append(
            f"{nume:<22}" + "".join(f"{metrici[m]:>12.3f}" for m in nume_metrici)
        )

    return "\n".join(linii)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fara-semantic", action="store_true",
                        help="ruleaza doar varianta pe cuvinte-cheie")
    parser.add_argument("--fara-hibrid", action="store_true",
                        help="omite metoda hibrida")
    parser.add_argument("--faiss", action="store_true",
                        help="foloseste indexare FAISS pentru cautarea semantica")
    argumente = parser.parse_args()

    faq = incarca_faq()
    intrebari_test = incarca_intrebari_test()
    print(f"Set de date: {len(faq)} intrebari, {len(intrebari_test)} intrebari de test\n")

    sisteme = {"Cuvinte-cheie (TF-IDF)": CautareCuvinteCheie(faq)}

    if not argumente.fara_semantic:
        from src.cautare_semantica import CautareSemantica
        print("Se incarca modelul de embeddings (prima rulare descarca ~120 MB)...")
        sistem_semantic = CautareSemantica(faq, foloseste_faiss=argumente.faiss)
        sisteme["Semantic (embeddings)"] = sistem_semantic

        # Citim starea reala a obiectului, nu argumentul primit, ca sa se vada
        # in output daca indexul chiar s-a construit.
        if sistem_semantic.index is not None:
            print(f"Cautare semantica: FAISS IndexFlatIP, "
                  f"{sistem_semantic.index.ntotal} vectori de dimensiune "
                  f"{sistem_semantic.index.d}")
        else:
            print("Cautare semantica: liniara (numpy), fara index")

        if not argumente.fara_hibrid:
            from src.cautare_hibrida import CautareHibrida
            sisteme["Hibrid (RRF)"] = CautareHibrida(
                sisteme["Cuvinte-cheie (TF-IDF)"], sistem_semantic
            )
        print()

    metrici_pe_sistem = {}
    detalii_pe_sistem = {}

    for nume, sistem in sisteme.items():
        metrici, detalii = evalueaza(sistem, intrebari_test)
        metrici_pe_sistem[nume] = metrici
        detalii_pe_sistem[nume] = detalii

    tabel = tabel_comparativ(metrici_pe_sistem)
    print(tabel + "\n")

    DIRECTOR_REZULTATE.mkdir(exist_ok=True)
    (DIRECTOR_REZULTATE / "metrici.txt").write_text(tabel + "\n", encoding="utf-8")

    # Aceleasi cifre in format JSON, ca sa le poata citi si interfata.
    with open(DIRECTOR_REZULTATE / "metrici.json", "w", encoding="utf-8") as f:
        json.dump(metrici_pe_sistem, f, ensure_ascii=False, indent=2)

    for nume, detalii in detalii_pe_sistem.items():
        corecte, gresite = separa_corecte_gresite(detalii)
        if "cheie" in nume:
            eticheta = "cuvinte_cheie"
        elif "Hibrid" in nume:
            eticheta = "hibrid"
        else:
            eticheta = "semantic"

        print(f"{nume}: {len(corecte)} corecte pe pozitia 1, {len(gresite)} gresite")

        if gresite:
            print("  Exemple de esecuri:")
            for d in gresite[:3]:
                pozitie = d["pozitie"] if d["pozitie"] else "negasit"
                print(f"    '{d['intrebare']}'")
                print(f"      asteptat: {d['id_asteptat']}, pozitie: {pozitie}")
                print(f"      returnat: {d['intrebare_top1']}")

        with open(DIRECTOR_REZULTATE / f"detalii_{eticheta}.json", "w",
                  encoding="utf-8") as f:
            json.dump({"corecte": corecte, "gresite": gresite}, f,
                      ensure_ascii=False, indent=2)
        print()

    print(f"Rezultatele au fost salvate in {DIRECTOR_REZULTATE}/")


if __name__ == "__main__":
    main()
