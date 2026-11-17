# Research references and attribution

PolyBot combines engineering choices informed by the work below. These are
conceptual references, not claims of complete reproductions or equivalent results.

| Work | Relation to this project |
| --- | --- |
| [Schulman et al., Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347) | Clipped on-policy optimization |
| [Cobbe et al., Phasic Policy Gradient (2020)](https://arxiv.org/abs/2009.04416) | Separate policy and auxiliary value phases |
| [Wang et al., PPG Reloaded (2023)](https://proceedings.mlr.press/v202/wang23aw.html) | Regularization and data-diversity experiments |
| [Silver et al., Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (2017)](https://arxiv.org/abs/1712.01815) | Policy/value search and self-play background |
| [Pérez-Liébana et al., Tribes: A New Turn-Based Strategy Game for AI Research (2020)](https://gaigresearch.github.io/afm/papers/TribesAIIDE20.pdf) | Turn structure and planning benchmarks in a Polytopia implementation |
| [Zhang, Building a Superhuman Polytopia Bot Using Self-Play RL (2026)](https://thekevinz.com/blog/polytopia-bot) | First-hand engineering report on a fogged value head and search through an opponent reply |
| [Cowling, Powley and Whitehouse, Information Set Monte Carlo Tree Search (2012)](https://eprints.whiterose.ac.uk/id/eprint/75048/) | Search with hidden information |
| [Brown et al., Combining Deep Reinforcement Learning and Search for Imperfect-Information Games (2020)](https://arxiv.org/abs/2007.13544) | ReBeL and value functions conditioned on public belief |
| [Schmid et al., Student of Games (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10651118/) | Learning and search with values conditioned on players' information states |
| [Willemsen, Baier and Kaisers, Value Targets in Off-Policy AlphaZero (2022)](https://research.tue.nl/files/217685150/s00521_021_05928_5.pdf) | Comparing value targets derived from different continuation policies |
| [Baisero and Amato, Unbiased Asymmetric Reinforcement Learning under Partial Observability (2022)](https://arxiv.org/abs/2105.11674) | Limitations of state-only critics with history-dependent policies |
| [Lanctot et al., A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning (2017)](https://arxiv.org/abs/1711.00832) | Population-learning background |
| [Balduzzi et al., Open-ended Learning in Symmetric Zero-sum Games (2019)](https://arxiv.org/abs/1901.08106) | Diversity and nontransitive populations |
| [Muller et al., A Generalized Training Approach for Multiagent Learning (2019)](https://arxiv.org/abs/1909.12823) | Broader population-learning background |
| [OpenAI et al., Dota 2 with Large Scale Deep Reinforcement Learning (2019)](https://arxiv.org/abs/1912.06680) | Large-scale self-play background |
| [DeepMind, AlphaStar: Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/) | League roles and opponent-sampling inspiration |

See [Training a value head for Polytopia](value_training.md) for Polytopia
experiments, value targets and search with hidden information.

[PolyEnv](https://github.com/FryOne2137/PolyEnv) supplies the game rules, legal
actions, simulation, detached belief-world construction and replay format.
Its [documentation](https://polyenv.readthedocs.io/en/latest/) and license are
maintained separately. PolyBot's diagrams are generated from this repository's
own Python source; proprietary game art is not bundled.

The project author is identified by the public handle **FryOne2137** in the
[MIT license](../LICENSE). Research citations for this software are described in
[CITATION.cff](../CITATION.cff).
