# -*- coding: utf-8 -*-
"""
=============================================================================
DRUG REPURPOSING PIPELINE USING KNOWLEDGE GRAPHS & NETWORK ANALYSIS
=============================================================================

PURPOSE:
    This pipeline identifies candidate drugs for diseases they were NOT
    originally designed to treat — a process called drug repurposing
    (or drug repositioning). It does this by:

      1. Fetching real drug–target bioactivity data from ChEMBL (a public
         pharmaceutical database).
      2. Constructing a heterogeneous knowledge graph linking:
             Drugs ──targets──► Genes/Proteins ──associated_with──► Diseases
      3. Applying a graph-based link prediction algorithm (Common Neighbors)
         to score and rank novel drug–disease associations.
      4. Exposing a simple query interface to retrieve ranked candidate drugs
         for any given disease.

DEPENDENCIES:
    - chembl_webresource_client  : Python client for the ChEMBL REST API
    - pandas                     : Tabular data handling
    - networkx                   : Graph construction and analysis
    - random                     : Reproducible random seeding

LIMITATIONS (for this prototype):
    - Disease associations are synthetically generated (not real clinical data).
    - The scoring method (Common Neighbors) is a simple heuristic; production
      systems use more sophisticated algorithms (e.g., matrix factorisation,
      graph neural networks).
    - Only IC50 bioactivity records are used as a proxy for drug–target binding.
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DEPENDENCY INSTALLATION
# ─────────────────────────────────────────────────────────────────────────────

# Install the official ChEMBL Python client (required in notebook environments).
!pip install chembl_webresource_client


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

from chembl_webresource_client.new_client import new_client   # ChEMBL API client
import pandas as pd                                            # DataFrame operations
import networkx as nx                                          # Graph data structures
import random                                                  # Reproducible sampling


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DATA ACQUISITION FROM ChEMBL
# ─────────────────────────────────────────────────────────────────────────────
# ChEMBL is a large open-access bioactivity database maintained by EMBL-EBI.
# IC50 (half-maximal inhibitory concentration) is a standard potency measure:
# a lower IC50 means a drug inhibits a target more effectively.

activity = new_client.activity   # Initialise the activity endpoint

data = []           # Will hold cleaned drug–target pairs
seen_drugs = set()  # Tracks unique drug IDs to cap the dataset size

# Fetch the first 500 IC50 activity records from ChEMBL.
# Each record links a molecule (drug) to a biological target (protein/gene).
res = activity.filter(standard_type="IC50")[:500]

for r in res:
    drug   = r.get('molecule_chembl_id')   # ChEMBL ID for the drug compound
    target = r.get('target_chembl_id')     # ChEMBL ID for the biological target

    if drug and target:
        if drug not in seen_drugs:
            seen_drugs.add(drug)           # Register each new drug once

        # Collect records until we have 30 unique drugs
        if len(seen_drugs) <= 30:
            data.append({"drug": drug, "target": target})

    # Early exit once 30 unique drugs have been gathered
    if len(seen_drugs) == 30:
        break

# Build a DataFrame and remove exact duplicate rows
df = pd.DataFrame(data).drop_duplicates()

# Keep at most 2 targets per drug to prevent highly connected drugs from
# dominating the graph and skewing link-prediction scores
df = df.groupby('drug').head(2)

print("Unique drugs:", df['drug'].nunique())
print(df.head())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: KNOWLEDGE GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
# A knowledge graph represents entities (drugs, genes, diseases) as NODES
# and biological relationships as labelled EDGES.
#
# Graph schema used here:
#
#   [Drug] ──"targets"──────────► [Gene/Target]
#   [Gene]  ──"associated_with"──► [Disease]
#   [Drug]  ──"treats"───────────► [Disease]   ← added then REMOVED for prediction

G = nx.Graph()   # Undirected graph (paths traversed in both directions)

# ── 4a. Add Drug Nodes ────────────────────────────────────────────────────────
drugs   = df['drug'].unique()
targets = df['target'].unique()

G.add_nodes_from(drugs,   type="drug")   # Node attribute 'type' enables filtering
G.add_nodes_from(targets, type="gene")

# ── 4b. Add Drug → Target Edges ───────────────────────────────────────────────
for _, row in df.iterrows():
    G.add_edge(row['drug'], row['target'], relation="targets")

# ── 4c. Add Synthetic Disease Nodes ───────────────────────────────────────────
# NOTE: In a production pipeline these would be real disease ontology nodes
# (e.g., from DisGeNET, OMIM, or the Disease Ontology) with curated
# gene–disease associations.
diseases = [f"Disease{i}" for i in range(1, 6)]
G.add_nodes_from(diseases, type="disease")

# ── 4d. Add Target → Disease Edges ────────────────────────────────────────────
# Cycle targets through diseases using modulo arithmetic to ensure every
# disease node receives at least one connected target.
random.seed(42)                            # Seed for reproducibility
target_list = list(targets)

for i, target in enumerate(target_list):
    disease = diseases[i % len(diseases)]  # Round-robin assignment
    G.add_edge(target, disease, relation="associated_with")

# ── 4e. Add Known Drug → Disease Edges (Ground Truth) ─────────────────────────
# For the first 5 drugs, trace one of their known targets to a connected
# disease and record that as a "known treatment" edge. These edges simulate
# ground-truth labels that will be hidden during prediction (see Section 5).
known_drugs = list(drugs)[:5]

for drug in known_drugs:
    neighbors = list(G.neighbors(drug))
    target    = random.choice(neighbors)   # Pick one connected target randomly

    for n in G.neighbors(target):
        if "Disease" in n:
            G.add_edge(drug, n, relation="treats")   # Known drug–disease link

print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))

for edge in list(G.edges(data=True))[:10]:
    print(edge)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: GRAPH PREPARATION — HIDE KNOWN LABELS (Link Prediction Setup)
# ─────────────────────────────────────────────────────────────────────────────
# To evaluate whether our algorithm can REDISCOVER known drug–disease links,
# we remove those edges from the graph before scoring. The algorithm must then
# infer them using only the remaining graph structure.
# This is the standard "held-out edge" evaluation protocol in link prediction.

edges_to_remove = [
    (u, v)
    for u, v, data in G.edges(data=True)
    if data.get("relation") == "treats"
]

G.remove_edges_from(edges_to_remove)
print("Removed known drug-disease edges")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: LINK PREDICTION — COMMON NEIGHBORS SCORING
# ─────────────────────────────────────────────────────────────────────────────
# Algorithm: Common Neighbors (CN)
#
#   score(drug, disease) = |N(drug) ∩ N(disease)|
#
# where N(x) is the set of graph neighbours of node x.
# Intuition: if a drug and a disease share many intermediate nodes (genes),
# it is likely the drug influences a pathway relevant to that disease.
#
# Higher-order alternatives used in research:
#   - Jaccard coefficient (normalised CN)
#   - Adamic-Adar index (down-weights high-degree hubs)
#   - Resource Allocation index
#   - Graph Neural Network embeddings (state-of-the-art)

drugs    = [n for n, d in G.nodes(data=True) if d['type'] == 'drug']
diseases = [n for n, d in G.nodes(data=True) if d['type'] == 'disease']

# Generate all (drug, disease) pairs that do NOT yet have a direct edge
candidates = [
    (drug, disease)
    for drug in drugs
    for disease in diseases
    if not G.has_edge(drug, disease)
]

scores = []

for drug, disease in candidates:
    common_neighbors = list(nx.common_neighbors(G, drug, disease))
    score = len(common_neighbors)   # Raw CN score

    if score > 0:
        scores.append((drug, disease, score, common_neighbors))

# Rank all candidate pairs by score in descending order
scores = sorted(scores, key=lambda x: x[2], reverse=True)

# Display the top 10 repurposing predictions
top_predictions = scores[:10]

print("\n=== TOP 10 DRUG REPURPOSING PREDICTIONS ===")
for drug, disease, score, neighbors in top_predictions:
    print(f"\nPrediction: {drug} → {disease}  (score={score})")
    for gene in neighbors:
        print(f"  Path: {drug} → {gene} → {disease}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: QUERY INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def query_drugs_for_disease(G, disease):
    """
    Retrieve and rank candidate drugs for a specified disease node.

    Uses the Common Neighbors heuristic to score every drug that does not
    already have a direct edge to the queried disease. Returns the top 5
    candidates along with the shared gene pathways that explain each prediction.

    Parameters
    ----------
    G       : nx.Graph  — The knowledge graph (drugs, genes, diseases)
    disease : str       — Target disease node identifier (e.g., "Disease1")

    Output
    ------
    Prints a ranked list of candidate drugs with supporting gene-level paths.
    """
    results = []
    drugs = [n for n, d in G.nodes(data=True) if d['type'] == 'drug']

    for drug in drugs:
        if not G.has_edge(drug, disease):   # Exclude already-known treatments
            common = list(nx.common_neighbors(G, drug, disease))
            if common:
                results.append((drug, len(common), common))

    # Sort by score descending, return top 5
    results = sorted(results, key=lambda x: x[1], reverse=True)

    print(f"\nQuery: Candidate drugs for {disease}")
    for drug, score, genes in results[:5]:
        print(f"\n  {drug}  (score={score})")
        for g in genes:
            print(f"    Path: {drug} → {g} → {disease}")


def agent(query):
    """
    Minimal natural-language query dispatcher.

    Parses a plain-English query string and routes it to the appropriate
    graph query function. Intended as a lightweight interface layer that
    could be extended with a full NLP parser or LLM backbone.

    Supported query patterns:
        "Find drugs for <DiseaseID>"   → calls query_drugs_for_disease()

    Parameters
    ----------
    query : str — Natural language query from the user

    Example
    -------
        agent("Find drugs for Disease1")
    """
    if "drugs for" in query.lower():
        disease = query.split()[-1]   # Extract the last token as the disease ID
        query_drugs_for_disease(G, disease)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: ENTRY POINT — EXAMPLE QUERY
# ─────────────────────────────────────────────────────────────────────────────

agent("Find drugs for Disease1")
