\# decisional-ai (dai)



\*\*decisional-ai\*\* é uma camada de \*decision-first AI\* que transforma probabilidades

(predict\_proba) em \*\*decisões defendíveis\*\* sob custo, capacidade e governança.



\## Baseline v0.9.0



Este repositório inicia com o \*\*baseline single-file estável\*\*:



\- decisional\_ai\_v0\_9\_full\_v5.py



\### Principais capacidades

\- Threshold ótimo por custo (best\_threshold)

\- Curva de decisão estruturada (decision\_curve)

\- Triagem 2-estágios (auto / review / defer)

\- Log operacional com id\_hash (sem PII)

\- Export de artefatos de auditoria



\## Uso rápido



python

from decisional\_ai\_v0\_9\_full\_v5 import Flow, DecisionConfig, batch\_decide

