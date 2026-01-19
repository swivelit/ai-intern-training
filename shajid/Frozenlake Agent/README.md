# ❄️ FrozenLake Agent (Q-Learning)

This project trains a **Tabular Q-Learning Reinforcement Learning agent** to solve the **FrozenLake-v1** environment using **OpenAI Gymnasium**.

The agent learns the best actions to reach the goal safely without falling into holes by updating a **Q-table** using the Q-learning update rule.

---

## 📌 Environment Details

- **Environment:** FrozenLake-v1
- **Library:** Gymnasium (OpenAI Gym)
- **State Space:** Discrete (16 states for 4x4 map)
- **Action Space:** Discrete (4 actions)
  - `0 = Left`
  - `1 = Down`
  - `2 = Right`
  - `3 = Up`

---

## ✅ Expected Output (Project Requirements)

✔ Train tabular Q-learning agent  
✔ Plot reward curve  
✔ Demonstrate solved environment  
✔ Submit Q-table + training code  

---

## 📂 Project Structure
    FrozenLake-Q-Learning/
    │
    ├── frozenlake_qlearning.py
    ├── q_table_frozenlake.npy
    └── README.md

## ⚙️ Installation

Install the required libraries using:

```bash
pip install gymnasium numpy matplotlib

##  If you want GUI rendering (render_mode="human"), install pygame also:

    pip install pygame
