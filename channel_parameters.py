import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as mticker # Import the ticker module

# Set Seaborn style
sns.set(style="whitegrid")

# Data including 512-channel results
channels = [32, 64, 128, 256, 384, 512]
acer = [0.471910, 0.389904, 0.407075, 0.192129, 0.317154, 0.326683]
auc = [0.509668, 0.636326, 0.609212, 0.888868, 0.698884, 0.714155]
acer = [x * 100 for x in acer]  # Convert to percentage
auc = [x * 100 for x in auc]  # Convert to percentage

# Create directory to save figures
save_dir = "channel_plots"
os.makedirs(save_dir, exist_ok=True)

# Plot AUC
plt.figure(figsize=(6, 5), dpi=200)
plt.plot(channels, auc, marker='o', color=sns.color_palette("deep")[0], label='AUC')
plt.xlabel('Number of Channels')
plt.ylabel('AUC (%)')

# --- MODIFICATION FOR ONE DECIMAL PLACE ---
# Get the current axes
ax_auc = plt.gca()
# Set the y-axis major tick formatter to display one decimal place
ax_auc.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
# ------------------------------------------

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "auc_vs_channels.png"))
plt.show()

# Plot ACER
plt.figure(figsize=(6, 5), dpi=200)
plt.plot(channels, acer, marker='o', color=sns.color_palette("deep")[1], label='ACER')
plt.xlabel('Number of Channels')
plt.ylabel('ACER (%)')

# --- MODIFICATION FOR ONE DECIMAL PLACE ---
# Get the current axes
ax_acer = plt.gca()
# Set the y-axis major tick formatter to display one decimal place
ax_acer.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
# ------------------------------------------

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "acer_vs_channels.png"))
plt.show()