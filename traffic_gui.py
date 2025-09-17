import tkinter as tk
from tkinter import ttk, messagebox, Scale
import threading
import folium
import webbrowser
import tempfile
import osmnx as ox
import networkx as nx
from geopy.geocoders import Nominatim
from itertools import islice
import time
import math
import random
from folium import plugins
import io
from PIL import Image, ImageTk
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# Set OSMnx to use a specific CRS that works well with folium
ox.settings.use_cache = True
ox.settings.log_console = True

# -----------------------------
# Utilities
# -----------------------------
geolocator = Nominatim(user_agent="dsa_route_app")

def geocode_candidates(place, limit=5):
    try:
        return geolocator.geocode(f"{place}, India", exactly_one=False, limit=limit) or []
    except Exception:
        return []

def select_candidate_dialog(root, title, candidates):
    if not candidates:
        return None
    sel = {"choice": None}
    dlg = tk.Toplevel(root)
    dlg.title(title)
    dlg.transient(root)
    dlg.grab_set()
    dlg.geometry("700x320")
    ttk.Label(dlg, text="Select the correct location from candidates:").pack(anchor="w", padx=8, pady=(8,0))
    listbox = tk.Listbox(dlg, width=120, height=12)
    listbox.pack(fill="both", expand=True, padx=8, pady=8)
    for c in candidates:
        desc = getattr(c, "address", None) or str(c)
        lat = getattr(c, "latitude", None)
        lon = getattr(c, "longitude", None)
        listbox.insert(tk.END, f"{desc}  |  lat:{lat:.6f} lon:{lon:.6f}")
    listbox.selection_set(0)
    btn_frame = ttk.Frame(dlg)
    btn_frame.pack(fill="x", padx=8, pady=(0,8))
    def on_ok():
        sel_idx = listbox.curselection()
        if not sel_idx:
            messagebox.showwarning("Select", "Choose a candidate or Cancel.")
            return
        sel["choice"] = candidates[sel_idx[0]]
        dlg.destroy()
    def on_cancel():
        dlg.destroy()
    ttk.Button(btn_frame, text="OK", command=on_ok).pack(side="right", padx=(0,8))
    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right")
    root.wait_window(dlg)
    return sel["choice"]

def choose_place(root, prompt, query_text):
    candidates = geocode_candidates(query_text, limit=7)
    if not candidates:
        raise ValueError(f"No geocoding candidates found for: {query_text}")
    chosen = select_candidate_dialog(root, prompt, candidates)
    if not chosen:
        raise ValueError("Location selection cancelled by user")
    return float(chosen.latitude), float(chosen.longitude)

def convert_to_simple_graph(G_multi):
    """
    Convert a MultiDiGraph from OSMnx to a simple weighted DiGraph
    by keeping the shortest edge between each pair of nodes.
    """
    G_simple = nx.DiGraph()
    for u, v, data in G_multi.edges(data=True):
        w = data.get("length", 1)
        if G_simple.has_edge(u, v):
            if w < G_simple[u][v]['weight']:
                G_simple[u][v]['weight'] = w
        else:
            G_simple.add_edge(u, v, weight=w)
    for n, data in G_multi.nodes(data=True):
        G_simple.add_node(n, **data)
    return G_simple

def ensure_wgs84(G):
    """Ensure the graph is in WGS84 (lat/lon) coordinate system"""
    if 'crs' in G.graph and G.graph['crs'] != 'epsg:4326':
        # Convert to WGS84 if not already
        G = ox.projection.project_graph(G, to_crs='epsg:4326')
    elif 'crs' not in G.graph:
        # Assume it's WGS84 if no CRS is specified
        G.graph['crs'] = 'epsg:4326'
    return G

def k_shortest_paths(G, start_point, end_point, k=3):
    # Ensure the graph is in WGS84
    G = ensure_wgs84(G)
    
    # Convert lat/lon to appropriate format for nearest_nodes
    start_node = ox.distance.nearest_nodes(G, X=start_point[1], Y=start_point[0])
    end_node = ox.distance.nearest_nodes(G, X=end_point[1], Y=end_point[0])
    
    # Get k shortest paths
    paths = list(islice(nx.shortest_simple_paths(G, start_node, end_node, weight="weight"), k))
    
    # Find a longer path by avoiding the shortest path edges
    if paths:
        # Create a copy of the graph
        G_long = G.copy()
        
        # Increase weight of edges in the shortest path to encourage a different route
        shortest_path = paths[0]
        for i in range(len(shortest_path) - 1):
            u, v = shortest_path[i], shortest_path[i + 1]
            if G_long.has_edge(u, v):
                G_long[u][v]['weight'] *= 5  # Make these edges much less attractive
        
        # Find a path that avoids the shortest path as much as possible
        try:
            longer_path = nx.shortest_path(G_long, start_node, end_node, weight="weight")
            paths.append(longer_path)
        except:
            # If we can't find a longer path, just use the longest of the k shortest
            paths.append(max(paths, key=lambda p: sum(G[u][v]['weight'] for u, v in zip(p[:-1], p[1:]))))
    
    return paths

def compute_path_distance(G, path):
    return sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:])) / 1000

def get_node_coordinates(G, node_id):
    """Extract coordinates from a node in a consistent way"""
    node_data = G.nodes[node_id]
    if 'y' in node_data and 'x' in node_data:
        return (node_data['y'], node_data['x'])
    elif 'lat' in node_data and 'lon' in node_data:
        return (node_data['lat'], node_data['lon'])
    else:
        # If coordinates aren't in the expected format, try to extract them
        # This might happen if the graph is in a different projection
        pos = (G.nodes[node_id].get('y', 0), G.nodes[node_id].get('x', 0))
        if pos != (0, 0):
            return pos
        # Last resort: use the node ID as coordinates (this is not ideal)
        return (node_id, node_id)

def calculate_bearing(pointA, pointB):
    """
    Calculate the bearing between two points.
    """
    lat1, lon1 = pointA
    lat2, lon2 = pointB
    
    dLon = lon2 - lon1
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    
    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing

# -----------------------------
# Simulation Classes
# -----------------------------
class TrafficLight:
    def __init__(self, node_id, position):
        self.node_id = node_id
        self.position = position
        self.state = "green"  # "red" or "green"
        self.manual_control = False  # If True, automatic cycling is disabled
        
    def toggle(self):
        self.state = "green" if self.state == "red" else "red"
        
    def set_green(self):
        self.state = "green"
        
    def set_red(self):
        self.state = "red"

class Vehicle:
    def __init__(self, vehicle_id, vehicle_type, path, graph, speed_kmh=40):
        self.id = vehicle_id
        self.type = vehicle_type  # "car" or "ambulance"
        self.path = path
        self.graph = graph
        self.speed = speed_kmh * 1000 / 3600  # convert to m/s
        self.current_segment = 0
        self.progress = 0  # 0 to 1 along current segment
        self.position = get_node_coordinates(graph, path[0])
        self.finished = False
        self.waiting_at_light = False
        self.color = "blue" if vehicle_type == "car" else "red"
        self.marker_size = 8 if vehicle_type == "car" else 10
        self.bearing = 0
        
    def update(self, traffic_lights, delta_time):
        if self.finished:
            return
            
        if self.current_segment >= len(self.path) - 1:
            self.finished = True
            return
            
        # Get current and next node
        current_node = self.path[self.current_segment]
        next_node = self.path[self.current_segment + 1]
        
        # Check if there's a traffic light at the next node
        next_traffic_light = traffic_lights.get(next_node)
        
        # If ambulance, it can change traffic lights to green
        if self.type == "ambulance" and next_traffic_light and next_traffic_light.state == "red":
            next_traffic_light.set_green()
        
        # If at a red light, wait
        if next_traffic_light and next_traffic_light.state == "red" and self.progress > 0.9:
            self.waiting_at_light = True
            return
            
        self.waiting_at_light = False
        
        # Calculate distance to travel in this update
        edge_length = self.graph[current_node][next_node]['weight']
        distance_to_travel = self.speed * delta_time
        
        # Update progress along current segment
        segment_progress = distance_to_travel / edge_length
        self.progress += segment_progress
        
        # If we've completed this segment, move to next one
        if self.progress >= 1:
            self.current_segment += 1
            self.progress = 0
            
            # Check if we've reached the destination
            if self.current_segment >= len(self.path) - 1:
                self.finished = True
                return
                
            # Update current and next nodes
            current_node = self.path[self.current_segment]
            next_node = self.path[self.current_segment + 1]
        
        # Calculate current position
        current_pos = get_node_coordinates(self.graph, current_node)
        next_pos = get_node_coordinates(self.graph, next_node)
        
        # Linear interpolation between nodes
        lat = current_pos[0] + (next_pos[0] - current_pos[0]) * self.progress
        lon = current_pos[1] + (next_pos[1] - current_pos[1]) * self.progress
        
        self.position = (lat, lon)
        
        # Calculate bearing for directional marker
        if self.current_segment < len(self.path) - 1:
            current_pos = get_node_coordinates(self.graph, self.path[self.current_segment])
            next_pos = get_node_coordinates(self.graph, self.path[self.current_segment + 1])
            self.bearing = calculate_bearing(current_pos, next_pos)

# -----------------------------
# GUI Application with Real-time Map
# -----------------------------
class DSAVisualApp:
    def __init__(self, root):
        self.root = root
        root.title("DSA Route Visualizer with Simulation")
        root.geometry("1400x800")
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # Left frame (inputs + table + buttons)
        self.left_frame = ttk.Frame(root, padding=12)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.left_frame.columnconfigure(1, weight=1)
        self.left_frame.rowconfigure(7, weight=1)

        ttk.Label(self.left_frame, text="Start Location:").grid(row=0, column=0, sticky="w")
        self.start_entry = ttk.Entry(self.left_frame, width=50)
        self.start_entry.grid(row=0, column=1, sticky="we", padx=6, pady=4)

        ttk.Label(self.left_frame, text="End Location:").grid(row=1, column=0, sticky="w")
        self.end_entry = ttk.Entry(self.left_frame, width=50)
        self.end_entry.grid(row=1, column=1, sticky="we", padx=6, pady=4)

        controls = ttk.Frame(self.left_frame)
        controls.grid(row=2, column=0, columnspan=2, sticky="we")
        self.search_btn = ttk.Button(controls, text="Find Routes", command=self.start_find_routes)
        self.search_btn.pack(side="left", padx=(0,8))
        self.status = ttk.Label(controls, text="Ready")
        self.status.pack(side="left", padx=12)

        cols = ("rank", "distance_km")
        self.tree = ttk.Treeview(self.left_frame, columns=cols, show="headings", height=10)
        self.tree.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=2, pady=6)
        self.tree.heading("rank", text="Route #")
        self.tree.column("rank", width=80, anchor="center")
        self.tree.heading("distance_km", text="Distance (km)")
        self.tree.column("distance_km", width=120, anchor="center")

        vsb = ttk.Scrollbar(self.left_frame, orient="vertical", command=self.tree.yview)
        vsb.grid(row=3, column=2, sticky="ns")
        self.tree.configure(yscrollcommand=vsb.set)

        # Simulation controls
        sim_frame = ttk.LabelFrame(self.left_frame, text="Simulation Controls", padding=8)
        sim_frame.grid(row=4, column=0, columnspan=2, sticky="we", pady=8)
        sim_frame.columnconfigure(0, weight=1)
        sim_frame.columnconfigure(1, weight=1)
        
        ttk.Label(sim_frame, text="Car Route:").grid(row=0, column=0, sticky="w")
        self.car_route = ttk.Combobox(sim_frame, values=[], state="readonly")
        self.car_route.grid(row=0, column=1, sticky="we", padx=4, pady=4)
        
        ttk.Label(sim_frame, text="Ambulance Route:").grid(row=1, column=0, sticky="w")
        self.ambulance_route = ttk.Combobox(sim_frame, values=[], state="readonly")
        self.ambulance_route.grid(row=1, column=1, sticky="we", padx=4, pady=4)
        
        ttk.Label(sim_frame, text="Simulation Speed:").grid(row=2, column=0, sticky="w")
        self.sim_speed = Scale(sim_frame, from_=1, to=10, orient="horizontal")
        self.sim_speed.set(5)
        self.sim_speed.grid(row=2, column=1, sticky="we", padx=4, pady=4)
        
        btn_frame = ttk.Frame(sim_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, sticky="we", pady=4)
        self.start_sim_btn = ttk.Button(btn_frame, text="Start Simulation", command=self.start_simulation)
        self.start_sim_btn.pack(side="left", padx=(0,8))
        self.stop_sim_btn = ttk.Button(btn_frame, text="Stop Simulation", command=self.stop_simulation, state="disabled")
        self.stop_sim_btn.pack(side="left")

        self.open_map_btn = ttk.Button(self.left_frame, text="Open Map for All Routes", command=self.open_all_routes_map)
        self.open_map_btn.grid(row=5, column=0, columnspan=2, sticky="we", pady=8)

        # Traffic light control panel
        self.light_control_frame = ttk.LabelFrame(self.left_frame, text="Traffic Light Controls", padding=8)
        self.light_control_frame.grid(row=6, column=0, columnspan=2, sticky="we", pady=8)
        
        # Right frame (map display)
        self.right_frame = ttk.Frame(root, padding=12)
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.rowconfigure(0, weight=1)
        self.right_frame.columnconfigure(0, weight=1)
        
        # Create matplotlib figure for map display
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Dijkstra explanation panel
        self.explanation_frame = ttk.Frame(self.left_frame)
        self.explanation_frame.grid(row=7, column=0, columnspan=2, sticky="nsew", pady=8)
        
        ttk.Label(self.explanation_frame, text="Dijkstra Path Explanation:").pack(anchor="w")
        self.text_panel = tk.Text(self.explanation_frame, wrap="word", height=10)
        self.text_panel.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(self.explanation_frame, orient="vertical", command=self.text_panel.yview)
        scrollbar.pack(side="right", fill="y")
        self.text_panel.configure(yscrollcommand=scrollbar.set)
        
        # Add theoretical explanation
        explanation = """
Dijkstra's Algorithm:

Dijkstra's algorithm is an algorithm for finding the shortest paths between nodes in a graph, which may represent, for example, road networks. It was conceived by computer scientist Edsger W. Dijkstra in 1956.

The algorithm works by building a set of nodes that have minimal distance from the source. It uses a priority queue to select the node with the current minimum distance, updates the distances of its neighbors, and repeats until the destination is reached.

K-Shortest Paths:

The k-shortest path algorithm finds not just the single shortest path, but the k shortest paths in increasing order of length. This is useful for considering alternative routes, which might be slightly longer but have other desirable properties.

In this application, we use Yen's algorithm to find the k-shortest paths. Yen's algorithm uses Dijkstra's algorithm as a subroutine.

Longer Path:

We also calculate a longer path by penalizing the edges of the shortest path, which forces the algorithm to find an alternative, typically longer route. This is useful for comparison purposes.

Simulation:

The simulation allows you to place a car and an ambulance on different routes. The ambulance has priority at traffic lights and can change them to green. Traffic lights can be manually controlled.
        """
        self.text_panel.insert("1.0", explanation)
        self.text_panel.config(state="disabled")  # Make it read-only

        self.cached_routes = []
        self.simulation_running = False
        self.simulation_thread = None
        self.traffic_lights = {}
        self.vehicles = {}
        self.route_lines = {}
        self.traffic_light_buttons = {}

    def set_status(self, txt):
        self.status.config(text=txt)
        self.root.update_idletasks()

    def start_find_routes(self):
        t = threading.Thread(target=self.find_routes, daemon=True)
        t.start()

    def find_routes(self):
        start_text = self.start_entry.get().strip()
        end_text = self.end_entry.get().strip()
        if not start_text or not end_text:
            messagebox.showwarning("Input needed", "Enter both start and end locations.")
            return

        self.search_btn.config(state="disabled")
        self.set_status("Resolving locations...")

        try:
            start_latlon = choose_place(self.root, "Select start location", start_text)
            end_latlon = choose_place(self.root, "Select end location", end_text)

            self.set_status("Building road network...")
            mid_lat = (start_latlon[0] + end_latlon[0]) / 2
            mid_lon = (start_latlon[1] + end_latlon[1]) / 2

            approx_dist = max(
                abs(start_latlon[0] - end_latlon[0]),
                abs(start_latlon[1] - end_latlon[1])
            ) * 111000 + 500
            approx_dist = min(approx_dist, 5000)  # max 5 km radius

            try:
                # Get graph with WGS84 CRS (latitude/longitude)
                G_multi = ox.graph_from_point(
                    (mid_lat, mid_lon), 
                    dist=approx_dist, 
                    network_type="drive", 
                    simplify=True
                )
                
                # Ensure the graph is in WGS84
                G_multi = ensure_wgs84(G_multi)
                
                # Convert to simple graph for pathfinding
                G = convert_to_simple_graph(G_multi)

                self.set_status("Computing paths...")
                paths = k_shortest_paths(G, start_latlon, end_latlon, k=3)
                self.cached_routes = [(G, path, compute_path_distance(G, path)) for path in paths]

                # Update table
                for i in self.tree.get_children():
                    self.tree.delete(i)
                
                route_options = []
                for idx, (_, _, dist) in enumerate(self.cached_routes):
                    if idx < 3:
                        route_name = f"P{idx+1} ({dist:.2f} km)"
                    else:
                        route_name = f"Longer Path ({dist:.2f} km)"
                    self.tree.insert("", "end", iid=str(idx), values=(route_name.split()[0], f"{dist:.2f}"))
                    route_options.append(route_name)
                
                # Update simulation route options
                self.car_route['values'] = route_options
                self.ambulance_route['values'] = route_options
                if route_options:
                    self.car_route.set(route_options[0])
                    self.ambulance_route.set(route_options[0])

                self.set_status("Done. Configure simulation and start.")

            except MemoryError:
                messagebox.showerror("Memory Error", "The selected area is too large. Try closer locations.")
                self.set_status("Error")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process map data: {str(e)}")
                self.set_status("Error")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.set_status("Error")
        finally:
            self.search_btn.config(state="normal")

    def open_all_routes_map(self):
        if not self.cached_routes:
            messagebox.showinfo("Select", "Find routes first.")
            return

        colors = ["blue", "green", "red", "purple"]
        # Use midpoint of first path for map centering
        G, path, distance = self.cached_routes[0]
        
        # Get coordinates for the path
        coords = [get_node_coordinates(G, n) for n in path]
        
        if not coords:
            messagebox.showerror("Error", "Could not extract coordinates from path")
            return
            
        mid = coords[len(coords)//2]
        m = folium.Map(location=mid, zoom_start=14)

        for idx, (G, path, distance) in enumerate(self.cached_routes):
            coords = [get_node_coordinates(G, n) for n in path]
            
            if coords:
                if idx < 3:
                    route_name = f"P{idx+1}"
                else:
                    route_name = "Longer Path"
                    
                folium.PolyLine(coords, color=colors[idx % len(colors)], weight=6, opacity=0.7,
                                tooltip=f"{route_name} Distance: {distance:.2f} km").add_to(m)
                folium.Marker(coords[0], icon=folium.Icon(color="green"), popup=f"{route_name} Start").add_to(m)
                folium.Marker(coords[-1], icon=folium.Icon(color="red"), popup=f"{route_name} End").add_to(m)

        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        m.save(tmp.name)
        webbrowser.open("file://" + tmp.name)
        self.set_status("Opened all routes in browser.")

    def setup_traffic_light_controls(self):
        """Create controls for each traffic light"""
        # Clear existing controls
        for widget in self.light_control_frame.winfo_children():
            widget.destroy()
            
        if not self.traffic_lights:
            ttk.Label(self.light_control_frame, text="No traffic lights in current simulation").pack()
            return
            
        # Create a scrollable frame for traffic light controls
        canvas = tk.Canvas(self.light_control_frame)
        scrollbar = ttk.Scrollbar(self.light_control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create a frame for each traffic light
        self.traffic_light_buttons = {}
        for i, (node_id, light) in enumerate(self.traffic_lights.items()):
            frame = ttk.Frame(scrollable_frame)
            frame.grid(row=i, column=0, sticky="we", pady=4)
            
            ttk.Label(frame, text=f"Light at Node {node_id}:").pack(side="left")
            
            # Toggle button
            btn = ttk.Button(frame, text=f"Set {'Red' if light.state == 'green' else 'Green'}",
                           command=lambda nid=node_id: self.toggle_traffic_light(nid))
            btn.pack(side="left", padx=5)
            
            # Status indicator
            status = ttk.Label(frame, text=f"Current: {light.state}", 
                             foreground="green" if light.state == "green" else "red")
            status.pack(side="left", padx=5)
            
            self.traffic_light_buttons[node_id] = (btn, status)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def toggle_traffic_light(self, node_id):
        """Toggle a traffic light state"""
        if node_id in self.traffic_lights:
            light = self.traffic_lights[node_id]
            light.toggle()
            
            # Update button text and status
            if node_id in self.traffic_light_buttons:
                btn, status = self.traffic_light_buttons[node_id]
                btn.config(text=f"Set {'Red' if light.state == 'green' else 'Green'}")
                status.config(text=f"Current: {light.state}", 
                            foreground="green" if light.state == "green" else "red")

    def start_simulation(self):
        if not self.cached_routes:
            messagebox.showinfo("Select", "Find routes first.")
            return
            
        car_route_idx = self.car_route.current()
        ambulance_route_idx = self.ambulance_route.current()
        
        if car_route_idx < 0 or ambulance_route_idx < 0:
            messagebox.showinfo("Select", "Select routes for both car and ambulance.")
            return
            
        # Get the selected routes
        G_car, car_path, car_dist = self.cached_routes[car_route_idx]
        G_ambulance, ambulance_path, ambulance_dist = self.cached_routes[ambulance_route_idx]
        
        # Create traffic lights at some nodes (every 3rd node for demonstration)
        self.traffic_lights = {}
        for i, node in enumerate(car_path):
            if i % 3 == 0:  # Place traffic lights at every 3rd node
                pos = get_node_coordinates(G_car, node)
                self.traffic_lights[node] = TrafficLight(node, pos)
        
        # Create vehicles
        self.vehicles = {
            "car": Vehicle("car", "car", car_path, G_car, speed_kmh=40),
            "ambulance": Vehicle("ambulance", "ambulance", ambulance_path, G_ambulance, speed_kmh=60)
        }
        
        # Setup traffic light controls
        self.setup_traffic_light_controls()
        
        # Start simulation thread
        self.simulation_running = True
        self.start_sim_btn.config(state="disabled")
        self.stop_sim_btn.config(state="normal")
        
        # Initialize the map display
        self.initialize_map_display(G_car, car_path, G_ambulance, ambulance_path)
        
        self.simulation_thread = threading.Thread(target=self.run_simulation, daemon=True)
        self.simulation_thread.start()

    def initialize_map_display(self, G_car, car_path, G_ambulance, ambulance_path):
        """Initialize the map display with routes"""
        self.ax.clear()
        
        # Plot car route
        car_coords = [get_node_coordinates(G_car, n) for n in car_path]
        car_lats, car_lons = zip(*car_coords) if car_coords else ([], [])
        self.ax.plot(car_lons, car_lats, 'b-', linewidth=2, label='Car Route')
        
        # Plot ambulance route
        ambulance_coords = [get_node_coordinates(G_ambulance, n) for n in ambulance_path]
        ambulance_lats, ambulance_lons = zip(*ambulance_coords) if ambulance_coords else ([], [])
        self.ax.plot(ambulance_lons, ambulance_lats, 'r-', linewidth=2, label='Ambulance Route')
        
        # Plot start and end points
        if car_coords:
            self.ax.plot(car_lons[0], car_lats[0], 'go', markersize=10, label='Start')
            self.ax.plot(car_lons[-1], car_lats[-1], 'ro', markersize=10, label='End')
        
        # Plot traffic lights
        for node_id, light in self.traffic_lights.items():
            pos = light.position
            color = 'green' if light.state == 'green' else 'red'
            self.ax.plot(pos[1], pos[0], 's', color=color, markersize=8, markeredgecolor='black')
        
        # Set labels and title
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_title('Route Simulation')
        self.ax.legend()
        self.ax.grid(True)
        
        # Set equal aspect ratio
        self.ax.set_aspect('equal', adjustable='datalim')
        
        self.canvas.draw()

    def update_map_display(self):
        """Update the map display with current vehicle positions and traffic light states"""
        if not self.simulation_running:
            return
            
        # Keep the base map (routes)
        self.ax.clear()
        
        # Replot the routes
        if "car" in self.vehicles:
            G_car = self.vehicles["car"].graph
            car_path = self.vehicles["car"].path
            car_coords = [get_node_coordinates(G_car, n) for n in car_path]
            car_lats, car_lons = zip(*car_coords) if car_coords else ([], [])
            self.ax.plot(car_lons, car_lats, 'b-', linewidth=2, alpha=0.5, label='Car Route')
        
        if "ambulance" in self.vehicles:
            G_ambulance = self.vehicles["ambulance"].graph
            ambulance_path = self.vehicles["ambulance"].path
            ambulance_coords = [get_node_coordinates(G_ambulance, n) for n in ambulance_path]
            ambulance_lats, ambulance_lons = zip(*ambulance_coords) if ambulance_coords else ([], [])
            self.ax.plot(ambulance_lons, ambulance_lats, 'r-', linewidth=2, alpha=0.5, label='Ambulance Route')
        
        # Plot start and end points
        if car_coords:
            self.ax.plot(car_lons[0], car_lats[0], 'go', markersize=10, label='Start')
            self.ax.plot(car_lons[-1], car_lats[-1], 'ro', markersize=10, label='End')
        
        # Plot traffic lights
        for node_id, light in self.traffic_lights.items():
            pos = light.position
            color = 'green' if light.state == 'green' else 'red'
            self.ax.plot(pos[1], pos[0], 's', color=color, markersize=8, markeredgecolor='black')
        
        # Plot vehicles with directional markers
        for vehicle_id, vehicle in self.vehicles.items():
            if not vehicle.finished:
                # Calculate arrow direction based on bearing
                dx = 0.0001 * math.cos(math.radians(vehicle.bearing))
                dy = 0.0001 * math.sin(math.radians(vehicle.bearing))
                
                # Plot vehicle position with directional arrow
                self.ax.plot(vehicle.position[1], vehicle.position[0], 'o', 
                           color=vehicle.color, markersize=vehicle.marker_size)
                self.ax.arrow(vehicle.position[1], vehicle.position[0], dx, dy, 
                            head_width=0.0002, head_length=0.0002, fc=vehicle.color, ec=vehicle.color)
        
        # Set labels and title
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_title('Route Simulation - Live View')
        self.ax.legend()
        self.ax.grid(True)
        
        # Set equal aspect ratio
        self.ax.set_aspect('equal', adjustable='datalim')
        
        self.canvas.draw()

    def stop_simulation(self):
        self.simulation_running = False
        self.start_sim_btn.config(state="normal")
        self.stop_sim_btn.config(state="disabled")

    def run_simulation(self):
        """Run the simulation with real-time updates"""
        last_time = time.time()
        
        while self.simulation_running and not all(v.finished for v in self.vehicles.values()):
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            # Update vehicles
            for vehicle in self.vehicles.values():
                vehicle.update(self.traffic_lights, delta_time)
            
            # Update the map display in the GUI thread
            self.root.after(0, self.update_map_display)
            
            # Control simulation speed
            time.sleep(1 / self.sim_speed.get())
        
        self.set_status("Simulation finished.")
        self.start_sim_btn.config(state="normal")
        self.stop_sim_btn.config(state="disabled")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DSAVisualApp(root)
    root.mainloop()