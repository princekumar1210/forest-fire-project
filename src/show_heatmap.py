"""
show_heatmap.py
Creates a fire risk heatmap using simulated coordinates and risk levels
"""

import folium
from folium.plugins import HeatMap
import random
import webbrowser
import os

def generate_fire_risk_heatmap():
    # Generate random coordinates (simulate forest areas)
    base_lat, base_lon = 15.5, 78.5  # approx. Central India
    risk_points = [
        (base_lat + random.uniform(-2.0, 2.0),
         base_lon + random.uniform(-2.0, 2.0),
         random.uniform(0, 1))  # 0 = low risk, 1 = high risk
        for _ in range(50)
    ]

    # Create map
    m = folium.Map(
        location=[base_lat, base_lon],
        zoom_start=6,
        tiles="Stamen Terrain",
        attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
    )

    # Add heatmap
    HeatMap([[lat, lon, risk] for lat, lon, risk in risk_points],
            radius=25, blur=15, max_zoom=6).add_to(m)

    # Save map in outputs folder inside project root
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    os.makedirs(output_dir, exist_ok=True)
    map_path = os.path.join(output_dir, "fire_risk_map.html")
    m.save(map_path)

    # Open in default browser
    webbrowser.open(f"file://{map_path}")

    print("✅ Fire risk heatmap generated and opened in browser!")

if __name__ == "__main__":
    generate_fire_risk_heatmap()
