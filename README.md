# DSA Route Visualizer & Traffic Simulation

## Overview
This project is a **route visualization and traffic simulation system** designed to demonstrate shortest path algorithms, k-shortest paths, and real-time vehicle movement on road networks in India.  
It combines **graph theory, geospatial data, and simulation** in an interactive Python GUI.

The project has **two main components**:

1. **Route Finder (Stable)**  
   - Uses **OSMnx** and **NetworkX** to build road networks and compute shortest paths.
   - Allows selecting **start and end locations** via geocoding with multiple candidate options.
   - Computes **k-shortest paths** and a longer alternative path.
   - Displays route distances in a **Tkinter table**.
   - Visualizes routes in **Folium interactive maps**.

2. **Traffic Simulation (Under Development)**  
   - Simulates **vehicle movement** on the computed routes using **Tkinter + Matplotlib**.
   - Supports **cars and ambulances**, with ambulances having priority at traffic lights.
   - **Traffic lights** can be manually toggled or automatically controlled by ambulances.
   - Real-time vehicle movement, directional bearings, and dynamic traffic light updates.
   - Adjustable **simulation speed**.

---

## Features

### Route Finder
- Location search with geocoding and candidate selection.
- Automatic **road network graph generation** around chosen points.
- Computes shortest, k-shortest, and longer alternative paths.
- Displays path distances in a **table**.
- View routes in **interactive Folium maps**.

### Traffic Simulation
- Vehicles follow precomputed routes.
- Ambulance priority at traffic lights.
- Real-time vehicle positions with directional markers.
- Dynamic traffic light control via GUI.
- Adjustable simulation speed.
- Live map display using **Matplotlib**.

> ⚠️ Simulation is **under development** and may have incomplete features.

---

## Dependencies

- `tkinter`
- `threading`
- `folium`
- `webbrowser`
- `tempfile`
- `osmnx`
- `networkx`
- `geopy`
- `itertools`
- `time`, `math`, `random`
- `PIL` (Pillow)
- `matplotlib`
- `numpy`
- `requests`

---

## Usage

### Route Finder
1. Run the application:
    ```bash
    python traffic_gui.py
    ```
2. Enter **Start Location** and **End Location**.
3. Select the correct location from candidate options.
4. Click **Find Routes**.
5. View route distances in the table.
6. Click **Open Map for All Routes** to see interactive maps.

### Traffic Simulation
1. After computing routes, select **Car Route** and **Ambulance Route**.
2. Adjust **Simulation Speed**.
3. Click **Start Simulation**.
4. Vehicles move along their routes in the live map.
5. Traffic lights can be toggled manually; ambulances can override them.
6. Click **Stop Simulation** to end the simulation.

---

## Project Structure

DSA-Route-Simulator/
│
├── traffic_gui.py # Main GUI application
├── README.md # Project documentation
├── requirements.txt # Python dependencies (to be generated)
└── assets/ # Optional: images, icons


---

## Notes & Future Work
- **Performance:** Large map areas may cause memory issues.
- **Simulation:** Improve traffic light logic, multiple vehicles, and collision handling.
- **UI Enhancements:** Interactive traffic light placement and route editing.
- **AI/ML Integration:** Predict congestion and optimize ambulance routes dynamically.
