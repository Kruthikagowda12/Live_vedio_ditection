import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\krithika KJ\Desktop\datamodel\output\crowd_data.csv")


max_crowd = df["count"].max()
min_crowd = df["count"].min()
avg_crowd = df["count"].mean()
total_frames = len(df)

print("Maximum Crowd:", max_crowd)
print("Minimum Crowd:", min_crowd)
print("Average Crowd:", round(avg_crowd, 2))
print("Total Frames:", total_frames)


plt.figure(figsize=(12,6))
plt.plot(df["frame"], df["count"])
plt.xlabel("Frame Number")
plt.ylabel("Number of People")
plt.title("Crowd Density Throughout Video")
plt.grid(True)
plt.savefig("crowd_density_graph.png")
plt.show()