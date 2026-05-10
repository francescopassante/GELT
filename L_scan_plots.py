import matplotlib.pyplot as plt
import numpy as np

# N = 1000
# D = 2
# channel_dimensions = [in_channels, 16, 32]
# lr = 1e-3
# splits = [0.7, 0.15, 0.15]

Ls = [4, 8, 12, 16, 20, 24, 28, 32]
test_losses_plaquettes = np.array(
    [
        [
            0.01717679,
            0.15430574,
            0.59761062,
            1.23729314,
            4.44196692,
            5.67420044,
            7.43264852,
            10.51627159,
        ],
        [
            0.0129726,
            0.17351418,
            0.5288353,
            1.2584252,
            3.92064381,
            7.92166319,
            8.25286283,
            12.16067047,
        ],
    ]
)
train_epochs_plaquettes = np.array(
    [
        [171.0, 240.0, 220.0, 244.0, 235.0, 245.0, 244.0, 162.0],
        [222.0, 195.0, 158.0, 226.0, 180.0, 211.0, 244.0, 240.0],
    ]
)

test_losses_links = np.array(
    [
        [
            16.61995583,
            70.14940643,
            147.33751221,
            192.79042664,
            443.58779297,
            486.53728027,
            764.34353027,
            1092.29003906,
        ],
        [
            14.6162137,
            63.01112671,
            157.8212616,
            262.17309875,
            455.67015381,
            525.73504028,
            780.90612183,
            1072.42370605,
        ],
    ]
)

train_epochs_links = np.array(
    [
        [20.0, 18.0, 22.0, 20.0, 23.0, 24.0, 33.0, 50.0],
        [28.0, 25.0, 27.0, 48.0, 25.0, 56.0, 43.0, 51.0],
    ]
)

# avg over all the scans:
test_loss_plaquette_avg = np.mean(test_losses_plaquettes, axis=0)
train_epochs_plaquette_avg = np.mean(train_epochs_plaquettes, axis=0)
test_loss_links_avg = np.mean(test_losses_links, axis=0)
train_epochs_links_avg = np.mean(train_epochs_links, axis=0)


plt.figure(figsize=(10, 5))
plt.plot(Ls, test_loss_links_avg, marker="o")
plt.xlabel("L")
plt.ylabel("Test Loss")
plt.title("Test Loss vs L (links)")
plt.grid(True)
plt.savefig("Test loss vs L_links.png")

plt.figure(figsize=(10, 5))
plt.plot(Ls, train_epochs_links_avg, marker="o")
plt.xlabel("L")
plt.ylabel("Train Epochs")
plt.title("Train Epochs vs L (links)")
plt.grid(True)
plt.savefig("Train epochs vs L_links.png")

plt.figure(figsize=(10, 5))
plt.plot(Ls, test_loss_plaquette_avg, marker="o")
plt.xlabel("L")
plt.ylabel("Test Loss")
plt.title("Test Loss vs L (plaquettes)")
plt.grid(True)
plt.savefig("Test loss vs L_plaquettes.png")

plt.figure(figsize=(10, 5))
plt.plot(Ls, train_epochs_plaquette_avg, marker="o")
plt.xlabel("L")
plt.ylabel("Train Epochs")
plt.title("Train Epochs vs L (plaquettes)")
plt.grid(True)
plt.savefig("Train epochs vs L_plaquettes.png")
