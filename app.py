from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route("/")
def dashboard():
    df = pd.read_csv("output/crowd_data.csv")

    max_people = df["count"].max()
    avg_people = df["count"].mean()
    min_people = df["count"].min()
    total_frames = len(df)
    peak_frame = df.loc[df["count"].idxmax(), "frame"]

    return render_template(
        "dashboard.html",
        max_people=max_people,
        avg_people=round(avg_people, 2),
        min_people=min_people,
        total_frames=total_frames,
        peak_frame=peak_frame
    )

if __name__ == "__main__":
    app.run(debug=True)