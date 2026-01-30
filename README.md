## Disclaimer

This repository contains an **independent implementation** of the PG-FFL algorithm proposed in an existing research paper.

This is **NOT the original implementation by the paper authors**.
The code is intended **only for learning, experimentation, and academic project purposes**.

All credit for the original algorithm, formulation, and ideas belongs to the authors of the referenced paper.

---

# PG-FFL: Policy Gradient–Based Fair Federated Learning

This repository provides an **educational / research-oriented implementation of PG-FFL**, a **policy-gradient-based fairness-aware federated learning framework**, built on top of the **FedAvg** algorithm.

The purpose of this implementation is to study how **reinforcement learning can learn client aggregation weights** in federated learning to balance:

* **Global accuracy**
* **Fairness across clients**, measured using the **Gini coefficient** of client accuracies

> ⚠️ **Project status**
> This repository is a **work-in-progress research prototype**.
> It currently supports **Fashion-MNIST**, **MLP models**, and **non-IID client splits**.

---

## 📌 Key Features

* ✔️ Federated Learning with a **FedAvg-style training loop**
* ✔️ **Policy Gradient (REINFORCE)** agent for learning aggregation weights
* ✔️ **Fairness-aware reward function** using Gini coefficient
* ✔️ Supports **IID and non-IID data partitioning** (via manual switches)
* ✔️ Modular project structure (data / FL / RL / models)

---

## 🧠 Method Overview

### Federated Learning

* Multiple clients train local models on private data
* A central server aggregates client models into a global model

### PG-FFL (Policy Gradient for Fair Federated Learning)

* **State**: Concatenation of participating client model parameters
* **Action**: Aggregation weights for each selected client, sampled from a Gaussian policy
* **Policy**: Neural network producing aggregation means
* **Reward**:

[
r_t = - \mu_t \cdot \log(G_t)
]

Where:

* ( \mu_t ): Average client validation accuracy
* ( G_t ): Gini coefficient of client accuracies

This reward formulation encourages **high accuracy** while maintaining **fairness across clients**.

---

## 📂 Project Structure

```
pg-ffl/
│
├── data/
│   └── dataset.py          # Dataset loading & IID / non-IID splits
│
├── fl/
│   ├── client.py           # Client-side local training
│   ├── server.py           # Server logic and aggregation
│   ├── aggregation.py      # Weighted aggregation utilities
│   └── fairness.py         # Gini coefficient computation
│
├── models/
│   └── mlp.py              # MLP model (Fashion-MNIST)
│
├── rl/
│   └── policy.py           # Policy network & REINFORCE update
│
├── train/
│   └── trainer.py          # Training and evaluation loops
│
├── main.py                 # Experiment entry point
└── README.md
```

---

## 📊 Dataset & Model

### Dataset

* **Fashion-MNIST**
* 10 classes, grayscale images (28×28)
* Downloaded automatically via `torchvision`

### Model

* Simple **3-layer MLP**
* Input: flattened 28×28 images
* Output: 10-class logits

---

## 🔀 Data Partitioning

Currently supported (via manual code switching):

* **IID split**
* **Non-IID split by class partitions**

  * Each client holds data from a subset of classes
  * Client label distributions may overlap or be disjoint

---

## ⚙️ How to Run

```bash
python main.py
```

Default configuration:

* Clients: 5
* Client fraction per round: 0.6
* Local epochs: 1
* Client optimizer: SGD
* Policy optimizer: Adam

---

## 📈 Output Metrics

Each federated round reports:

* Average client validation accuracy
* Gini coefficient (fairness metric)
* Global test accuracy

Example output:

```
Round 10: Avg client accuracy = 0.9645, Gini = 0.0143, Global test acc = 0.3625
```

---

## 🚧 Known Limitations

* Dataset and model are **tightly coupled**
* Switching between **FedAvg and PG-FFL** requires manual code changes
* Policy state dimension scales poorly with model size
* Non-IID performance is **unstable and under investigation**

These limitations are intentional at the current research stage.

---

## 🧪 Research Intent

This repository is intended for:

* Understanding fairness-aware federated learning
* Studying RL-based aggregation strategies
* Academic experimentation (not production use)

---

## 🔮 Future Work (Not Implemented)

* Config-driven experiment setup
* Dataset-agnostic pipelines (CIFAR-10 / CIFAR-100)
* Strategy abstraction (FedAvg vs PG-FFL)
* Logging and plotting utilities
* Reproducibility scripts

---

## 📄 Paper Reference

This implementation is inspired by recent research on **policy-gradient-based fair federated learning**.

Please refer to the original paper for formal definitions, theoretical analysis, and experimental benchmarks.

---

## 📜 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

---

## 🙌 Acknowledgement

This work is inspired by research on **fair federated learning using deep reinforcement learning**.

> This implementation prioritizes **clarity and interpretability** over optimization.
