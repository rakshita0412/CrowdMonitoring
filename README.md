Crowd Monitoring App with alerts

A real-time crowd counting system using CSRNet that sends email alerts when crowd size exceeds a configurable threshold. The app visualizes crowd density with heatmaps and provides informative HTML email alerts with plots.  

Features

- Upload images (jpg, jpeg, png) to estimate crowd count.
- Heatmap overlay showing crowd density on the image.
- Sends HTML email alerts with heatmap and comparison plots.
- Configurable crowd threshold for alerts.
- Fully in-memory processing (no disk writes).

Demo

![App Screenshot](docs/demo_screenshot1.png)
![App Screenshot](docs/demo_screenshot2.png)
![Email Screenshot](docs/demo_screenshot3.png)
![Email Screenshot](docs/demo_screenshot4.png)



Installation

```bash
git clone https://github.com/rakshita0412/CrowdMonitoring.git
cd CrowdMonitoring
pip install -r requirements.txt
streamlit run src/app.py
