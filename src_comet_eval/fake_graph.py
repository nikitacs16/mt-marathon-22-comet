#!/usr/bin/env python3

import matplotlib.pyplot as plt

data_acc = [0.7, 0.5, 0.6, 0.62]
data_corr = [0.4, 0.3, 0.28, 0.25]

KWARGS = {"marker": "."}

plt.figure(figsize=(5,3))

plt.plot(
    data_acc,
    label="Challenge set accuracy",
    **KWARGS
)
plt.plot(
    data_corr,
    label="MQM correlation (Pearson)",
    **KWARGS
)

plt.ylabel("Performance (acc/corr)")
plt.xlabel("$\\alpha$ in DA-$\\alpha$ augmentation for $h^-$")
plt.legend()
plt.tight_layout()
plt.show()