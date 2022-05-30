# Tagparse

A tagger and a parser that can be trained separately or jointly.

The parser is a shallow version of Dozat & Manning's 2017 deep biaffine attention parser, as devised by Glavaš and Vulić (2021b).

The parser implementation builds on [TowerParse](https://github.com/codogogo/towerparse) (Glavaš and Vulić, 2021a). Credit is given in the source code where relevant.

## Bibliography

- Dozat, T. and C. D. Manning. 2017. Deep Biaffine Attention for Neural Dependency Parsing. Proceedings of ICLR 2017. <https://openreview.net/forum?id=Hk95PK9le>.
- Glavaš, G. and I. Vulić. 2021a. Climbing the Tower of Treebanks: Improving Low-Resource Dependency Parsing via Hierarchical Source Selection. Findings of ACL-IJCNLP 2021, pp. 4878–4888. <https://dx.doi.org/10.18653/v1/2021.findings-acl.431>
- Glavaš, G. and I. Vulić. 2021b. Is Supervised Syntactic Parsing Beneficial for Language Understanding Tasks? An Empirical Investigation. Proceedings of ACL, pp. 3090–3104. <https://dx.doi.org/10.18653/v1/2021.eacl-main.270>.