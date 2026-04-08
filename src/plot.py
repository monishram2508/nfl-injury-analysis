import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

root=Path(__file__).parent.parent
csv_path = root/"motion_data.csv"

df = pd.read_csv(csv_path)

df = df.dropna()

plt.figure()

for (injury, angle), group in df.groupby(['injury_id', 'angle_id']):
    plt.plot(group['frame_index'], group['velocity'], label=f"{injury}-{angle}")

plt.xlabel("Frame Index")
plt.ylabel("Velocity")
plt.title("Velocity vs Frame")
plt.legend()

plt.show()

plt.figure()

for (injury, angle), group in df.groupby(['injury_id', 'angle_id']):
    plt.plot(group['frame_index'], group['acceleration'], label=f"{injury}-{angle}")

plt.xlabel("Frame Index")
plt.ylabel("Acceleration")
plt.title("Acceleration vs Frame")
plt.legend()

plt.show()