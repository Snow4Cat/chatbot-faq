"""Calculul metricilor: recall@K si MRR."""


def evalueaza(sistem, intrebari_test, valori_k=(1, 3, 5)):
    """Ruleaza sistemul pe setul de test si calculeaza metricile.

    recall@K = cate intrebari au raspunsul corect in primele K rezultate.
    MRR = media lui 1/pozitie; scade daca raspunsul corect apare mai jos.
    """
    k_maxim = max(valori_k)
    detalii = []

    for test in intrebari_test:
        rezultate = sistem.cauta(test["intrebare"], k=k_maxim)
        ids = [r["id"] for r in rezultate]

        pozitie = ids.index(test["id_asteptat"]) + 1 if test["id_asteptat"] in ids else None

        detalii.append({
            "intrebare": test["intrebare"],
            "id_asteptat": test["id_asteptat"],
            "ids_returnate": ids,
            "pozitie": pozitie,
            "scor_top1": rezultate[0]["scor"],
            "intrebare_top1": rezultate[0]["intrebare"],
        })

    total = len(detalii)
    metrici = {
        f"recall@{k}": sum(
            1 for d in detalii if d["pozitie"] is not None and d["pozitie"] <= k
        ) / total
        for k in valori_k
    }
    metrici["MRR"] = sum(
        1 / d["pozitie"] for d in detalii if d["pozitie"] is not None
    ) / total

    return metrici, detalii


def separa_corecte_gresite(detalii):
    """Imparte rezultatele in corecte pe pozitia 1 si restul."""
    corecte = [d for d in detalii if d["pozitie"] == 1]
    gresite = [d for d in detalii if d["pozitie"] != 1]
    return corecte, gresite
