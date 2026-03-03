from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("model/CrowdHuman.pt")
cap = cv2.VideoCapture("video/MumbaiMetro_DemoCCTVClip_1.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

frame_id = 0
x_data = []
y_data = []

plt.ion()  # Interactive mode ON
fig, ax = plt.subplots()
line, = ax.plot([], [])

ax.set_xlabel("Frame Number")
ax.set_ylabel("Crowd Count")
ax.set_title("Live Crowd Density Graph")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    count = len(results[0].boxes)

    # Show count on video
    cv2.putText(frame, f'Count: {count}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Live Crowd Detection", frame)

    # Update graph
    x_data.append(frame_id)
    y_data.append(count)

    line.set_xdata(x_data)
    line.set_ydata(y_data)

    ax.relim()
    ax.autoscale_view()

    plt.draw()
    plt.pause(0.001)

    frame_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()