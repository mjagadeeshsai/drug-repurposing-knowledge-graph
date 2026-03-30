# Knowledge Graph–Driven Drug Repurposing

## Problem

This project explores how knowledge graphs and graph-based machine learning can be used to identify potential drug repurposing opportunities. The goal is to model relationships between drugs, biological targets (genes/proteins), and diseases, and to generate biologically meaningful and interpretable therapeutic hypotheses.

---

## Approach

* Constructed a biomedical knowledge graph using:

  * Drug–target interactions sourced from ChEMBL
  * Target–disease associations (synthetic for prototyping purposes)

* Represented drugs, genes, and diseases as nodes, and their biological relationships as edges to preserve relational structure

* Applied graph-based link prediction (common neighbors) to identify potential drug–disease relationships

* Built an agentic query layer to:

  * Accept natural queries (e.g., “Find drugs for DiseaseX”)
  * Execute graph-based reasoning
  * Return interpretable predictions with biological paths

---

## Example Output

Prediction: CHEMBL123 → Disease2

Reasoning:
CHEMBL123 → TargetA → Disease2
CHEMBL123 → TargetB → Disease2

This suggests that the drug may influence the disease through shared biological targets, providing a plausible mechanism for therapeutic effect.

---

## Agentic Query Example

Query: "Find drugs for Disease1"

Output:
CHEMBL456 (score=2)
Path: CHEMBL456 → TargetX → Disease1

CHEMBL789 (score=1)
Path: CHEMBL789 → TargetY → Disease1

The agent retrieves candidate drugs and explains predictions via interpretable graph paths.

---

## Key Insight

This system is designed to generate interpretable, biologically plausible therapeutic hypotheses by leveraging relational structure in biomedical data rather than relying solely on black-box predictive models.

---

## Limitations

* Target–disease relationships are simplified for prototyping
* No mechanism-of-action (MoA) weighting or pharmacological context
* Graph incompleteness may limit prediction accuracy
* Uses simple topological link prediction instead of advanced graph embeddings

---

## Future Work

* Integrate real-world disease association datasets (e.g., DisGeNET)
* Incorporate mechanism-of-action and pathway-level information
* Apply graph embeddings (Node2Vec) or Graph Neural Networks
* Extend agentic layer with LLM-based reasoning for hypothesis refinement

---

## Tech Stack

* Python
* NetworkX
* Pandas
* ChEMBL Web Resource Client

---

## Summary

This project demonstrates how knowledge graph–based reasoning can be applied to drug discovery, enabling the generation of explainable and testable hypotheses for drug repurposing. It reflects a shift from purely predictive modeling toward interpretable, biologically grounded machine learning systems.
