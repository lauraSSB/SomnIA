"""exp1_brow.py — H3 en validación cruzada: mejor config con y sin ceja-ojo, mismos 5 pliegues x 3 semillas."""
import json, itertools, csv
import ri_common as C
import exp1_configuraciones as E1

if __name__ == "__main__":
    from multiprocessing import Pool
    d = C.load_data()
    fo = json.load(open(E1.OUT / "pliegues.json"))["pliegues"]
    b = json.load(open(E1.OUT / "mejor_config.json"))
    jobs = [(b["n_conjuntos"], b["forma"], b["init"], k, s, ["ear", "mar", "brow"], fo[k])
            for k, s in itertools.product(range(5), E1.SEEDS)]
    with Pool(10) as p:
        rows = p.map(E1.run_one, jobs)
    with open(E1.OUT / "corridas_brow.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("ok", len(rows))
