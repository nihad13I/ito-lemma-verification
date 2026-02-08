import numpy as np
import matplotlib.pyplot as plt

# 1. Define Parameters
S0 = 100        # Initial Stock Price
mu = 0.05       # 5% Annual Drift
sigma = 0.2     # 20% Annual Volatility
T = 1.0         # 1 Year time horizon
dt = 0.001      # High resolution time step (1/1000th of a year)
n_steps = int(T / dt)

# 2. Simulation Setup
S = np.zeros(n_steps)
S[0] = S0
log_S_ito = np.zeros(n_steps)
log_S_ito[0] = np.log(S0)

# 3. Euler-Maruyama Loop
for t in range(1, n_steps):
    dW = np.random.normal(0, np.sqrt(dt)) # Brownian increment

    # Update Stock Price S_t (GBM)
    S[t] = S[t-1] + mu*S[t-1]*dt + sigma*S[t-1]*dW

    # Update Log-Price using Ito's Result directly
    # d(ln S) = (mu - 0.5 * sigma^2)dt + sigma * dW
    log_S_ito[t] = log_S_ito[t-1] + (mu - 0.5 * sigma**2)*dt + sigma*dW

# 4. Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(S, label='Simulated Stock Price ($S_t$)')
plt.title('GBM Path')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.log(S), label='ln($S_t$) from Simulation', alpha=0.7)
plt.plot(log_S_ito, label="Ito's Formula Prediction", linestyle='--')
plt.title('Numerical Verification of Ito Lemma')
plt.legend()
plt.show()
