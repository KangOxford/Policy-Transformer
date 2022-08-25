# Policy Transformer
Transformer in Reinforcement Learning
* casts the problem of RL as conditional sequence modeling
* comparison
    * traditional: fit value functions or compute policy gradients
    * transformer: output the optimal actions by a transformer
* By conditioning an autoregressive model on the desired return (reward), past states, and actions, the Decision Transformer model can generate future actions that achieve the desired return.
