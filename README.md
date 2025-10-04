# 🧠 Decentralized P2P Federated Learning

## 📘 Overview
This project implements a **fully decentralized peer-to-peer (P2P) federated learning system** that enables secure and privacy-preserving model training across multiple clients without relying on a central server.  
Each client trains its model locally on its private dataset and communicates with other peers via **custom TCP socket connections**, exchanging only model weights or logits — not raw data.

---

## 🚀 Key Features
- **Fully Decentralized P2P Architecture:**  
  Clients communicate directly using TCP sockets without any centralized coordinator.

- **Knowledge Distillation (KD):**  
  Utilized both **feature-based** and **response-based** knowledge distillation techniques to improve model accuracy while preserving privacy.

- **Data Privacy:**  
  Only model weights or logits are shared between peers — no raw data transmission.

- **Non-IID Data Distribution:**  
  Data is distributed across clients using **Dirichlet distribution**, ensuring realistic non-identically distributed datasets.

- **Multi-threaded Communication:**  
  TCP socket-based message passing is implemented using Python’s **multithreading** to support concurrent model updates.

- **Model Performance:**  
  Improved CNN model accuracy by **28%** through effective distillation and collaboration between clients.

---

## 🧩 Tech Stack
- **Language:** Python  
- **Frameworks/Libraries:** PyTorch, Scikit-learn  
- **Networking:** TCP Sockets  
- **Concurrency:** Multi-threading  
- **Datasets:** MNIST, CIFAR    

---

## 🏗️ System Architecture
The system simulates **30 clients**, each training one of the three CNN models:
- **CNN3**
- **CNN5**
- **CNN7**

Each client:
1. Receives a **non-IID subset** of data via Dirichlet distribution.  
2. Trains its local CNN on the assigned data.  
3. Shares either **logits** or **model weights** with other peers over TCP connections.  
4. Updates its local model using **knowledge distillation** techniques.

---

---

## ⚙️ FOR QUICK DEMO

1. **Clone the repository and run it as below:**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   pip install -r requirements.txt
   python main.py

---

## 📈 Results
- **Accuracy improved by 28% after applying knowledge distillation.
- **Demonstrated effective collaboration among 30 decentralized clients.
- **Preserved complete data privacy with no data sharing.

## 🔐 Privacy & Security

- **No central server: Eliminates single point of failure.
- **Weight/logit exchange only: Prevents raw data leakage.
- **Peer authentication: Ensures only trusted clients participate.

