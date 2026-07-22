# srcML Detection Models

Detects AI-generated vs human-written code using structural representations derived from [srcML](https://www.srcml.org/).

---

## Data Collection (`data/prepare.py`)

**Input:** HMCorp `.jsonl` — each row has a `code`/`contrast` pair (one human, one AI) with an `index` and `label`.

**Steps:**
1. Writes each code snippet to a `.java` file under `temp/`
2. Runs `srcml` to produce a `.xml` file per snippet
3. Strips the `filename` attribute from the XML to prevent label leakage
4. Filters out any rows where the XML still contains label-leaking keywords (`human`)
5. Merges everything into a single CSV with columns: `id`, `code`, `label`, `language`, `xml`

```bash
python -m model.metrics.srcml.data.prepare --input train.jsonl --output data/hmcorp_xml.csv
```

---

## Shared Embedder (`embedding/`)

All variants use **CodeT5+ 110M** (`Salesforce/codet5p-110m-embedding`) to embed text into 256-dim vectors. Accepts raw code, AST sequences, or srcML XML as input.

---

## Model Variants

### `xml/` — SVM on separate embeddings

Embeds each representation independently, concatenates, trains an RBF-SVM.

Feature importance via block permutation (shuffle each 256-dim block, measure accuracy drop).

| Combo | Dim | Acc | P | R | F1 | AUC | feat. importance |
|---|---|---|---|---|---|---|---|
| `code+xml` | 512 | 0.9509 | 0.9552 | 0.9463 | 0.9507 | 0.9894 | code 0.337, xml 0.179 |
| `code+ast+xml` | 768 | 0.9582 | 0.9624 | 0.9537 | 0.9580 | 0.9915 | code 0.272, ast 0.138, xml 0.126 |

```bash
sbatch model/metrics/srcml/xml/train.sh
```

---

### `graph/` — RandomForest on SCG + embeddings

Converts srcML XML into a Structural Code Graph (SCG) with hierarchical, sequential, and self-loop edges, then extracts 12 graph features.

**Input:** 256-dim code embedding + 12 SCG features = 268-dim vector  
**Classifier:** RandomForest (100 trees)  
**Feature importance:** native RF `feature_importances_`, split between embedding block and each SCG feature.

**SCG features:** `num_nodes`, `num_edges`, `num_self_loops`, `graph_density`, `num_cycles`, degree stats (sum/avg/max/min/median/in/out)

```bash
sbatch model/metrics/srcml/graph/train.sh
```

---

## Structure

```
srcml/
├── data/prepare.py       # data collection + XML generation
├── embedding/            # shared CodeT5+ embedder
├── xml/
│   ├── train.py          # SVM pipeline
│   └── train.sh          # SLURM job script
└── graph/
    ├── train.py          # RandomForest + SCG pipeline
    └── train.sh          # SLURM job script
```
