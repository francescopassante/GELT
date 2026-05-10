import matplotlib.pyplot as plt

# N = 1000
# D = 2
# channel_dimensions = [in_channels, 16, 32]
# lr = 1e-3
# splits = [0.7, 0.15, 0.15]

Ls = [4, 8, 12, 16, 20, 24, 28, 32]
test_losses = [
    0.01717679,
    0.15430574,
    0.59761062,
    1.23729314,
    4.44196692,
    5.67420044,
    7.43264852,
    10.51627159,
]
train_epochs = [171.0, 240.0, 220.0, 244.0, 235.0, 245.0, 244.0, 162.0]

plt.figure(figsize=(10, 5))
plt.plot(Ls, test_losses, marker="o")
plt.xlabel("L")
plt.ylabel("Test Loss")
plt.title("Test Loss vs L")
plt.grid(True)
plt.savefig("Test loss vs L")

plt.figure(figsize=(10, 5))
plt.plot(Ls, train_epochs, marker="o")
plt.xlabel("L")
plt.ylabel("Train Epochs")
plt.title("Train Epochs vs L")
plt.grid(True)
plt.savefig("Train epochs vs L")
