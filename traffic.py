import pygame
import heapq

# ------------------------- DIJKSTRA ALL SHORTEST PATHS -------------------------
def dijkstra_all_paths(graph, start, end):
    dist = {node: float('inf') for node in graph}
    parents = {node: [] for node in graph}
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        for v in graph[u]:
            if dist[v] > d + 1:
                dist[v] = d + 1
                parents[v] = [u]
                heapq.heappush(pq, (dist[v], v))
            elif dist[v] == d + 1:
                parents[v].append(u)

    all_paths = []

    def backtrack(node, path):
        if node == start:
            all_paths.append(path[::-1])
            return
        for p in parents[node]:
            backtrack(p, path + [p])

    backtrack(end, [end])
    return all_paths

# ------------------------- VEHICLE -------------------------
class Vehicle:
    def __init__(self, vid, vtype, path, coords):
        self.id = vid
        self.vtype = vtype
        self.path = path
        self.index = 0
        self.progress = 0.0
        self.arrived = False
        self.coords = coords
        self.speed = 0.02 if vtype == "car" else 0.04

    def update(self, traffic_lights):
        if self.arrived or self.index >= len(self.path) - 1:
            self.arrived = True
            return

        src = self.path[self.index]
        dst = self.path[self.index + 1]

        if dst in traffic_lights and traffic_lights[dst] == "RED" and self.vtype == "car":
            return

        self.progress += self.speed
        if self.progress >= 1.0:
            self.index += 1
            self.progress = 0.0
            if self.index >= len(self.path) - 1:
                self.arrived = True

    def current_position(self):
        if self.arrived:
            return self.coords[self.path[-1]]
        src = self.coords[self.path[self.index]]
        dst = self.coords[self.path[self.index + 1]]
        x = src[0] + (dst[0] - src[0]) * self.progress
        y = src[1] + (dst[1] - src[1]) * self.progress
        return (int(x), int(y))

# ------------------------- GRID GRAPH -------------------------
def build_grid_graph(rows, cols):
    graph = {}
    coords = {}
    WIDTH, HEIGHT = 600, 600
    MARGIN = 80
    NODE_RADIUS = 20
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            coords[node] = (MARGIN + c * ((WIDTH - 2*MARGIN)//(cols-1)),
                            MARGIN + r * ((HEIGHT - 2*MARGIN)//(rows-1)))
            graph[node] = []
            if r > 0:
                graph[node].append((r-1)*cols + c)
            if r < rows-1:
                graph[node].append((r+1)*cols + c)
            if c > 0:
                graph[node].append(r*cols + (c-1))
            if c < cols-1:
                graph[node].append(r*cols + (c+1))
    return graph, coords

# ------------------------- SIMULATION -------------------------
def run():
    pygame.init()
    WIDTH, HEIGHT = 900, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Traffic Simulation with Multiple Paths")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    big_font = pygame.font.SysFont(None, 28)

    ROWS, COLS = 4, 4
    graph, coords = build_grid_graph(ROWS, COLS)
    traffic_lights = {n:"GREEN" for n in coords}
    vehicles = []

    start_node, end_node = None, None
    all_paths = []
    chosen_path = []
    spawn_id = 0
    concept_text = "Click first node as START and second node as END."

    running = True
    while running:
        screen.fill((30,30,30))

        # Draw edges
        for u in graph:
            for v in graph[u]:
                if u < v:
                    pygame.draw.line(screen, (200,200,200), coords[u], coords[v], 2)

        # Highlight all paths
        for i, path in enumerate(all_paths):
            color = (255, 255, 0) if path != chosen_path else (50,150,255)
            for j in range(len(path)-1):
                pygame.draw.line(screen, color, coords[path[j]], coords[path[j+1]], 4 if path==chosen_path else 2)
            # Label path
            mid_node = path[len(path)//2]
            label = font.render(f"P{i+1}", True, (255,255,255))
            screen.blit(label, (coords[mid_node][0]+5, coords[mid_node][1]+5))

        # Draw nodes
        for node, pos in coords.items():
            color = (0,200,0)
            if node == start_node:
                color = (0,0,255)
            elif node == end_node:
                color = (255,140,0)
            if traffic_lights[node]=="RED":
                pygame.draw.circle(screen, (200,0,0), pos, 15)
            else:
                pygame.draw.circle(screen, color, pos, 15)
            label = font.render(str(node), True, (255,255,255))
            screen.blit(label, (pos[0]-8,pos[1]-8))

        # Update + draw vehicles
        for v in vehicles:
            v.update(traffic_lights)
            pos = v.current_position()
            color = (255,0,0) if v.vtype=="ambulance" else (0,150,255)
            pygame.draw.circle(screen, color, pos, 8)

        # Instructions & concept
        instructions = [
            "Controls:",
            "Click first node = START",
            "Click second node = END",
            "C - Spawn Car (first path)",
            "A - Spawn Ambulance (first path)",
            "T - Toggle Traffic Lights",
            "Q - Quit"
        ]
        y = 20
        for line in instructions:
            txt = font.render(line, True, (255,255,255))
            screen.blit(txt, (650,y))
            y+=25

        # Concept text
        concept_display = big_font.render(concept_text, True, (255,255,0))
        screen.blit(concept_display, (30, HEIGHT-50))

        pygame.display.flip()
        clock.tick(60)

        # Event handling
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
            elif event.type==pygame.MOUSEBUTTONDOWN:
                pos_click = pygame.mouse.get_pos()
                for node, pos in coords.items():
                    if (pos[0]-pos_click[0])**2 + (pos[1]-pos_click[1])**2 <= 15**2:
                        if start_node is None:
                            start_node = node
                            concept_text = f"Start node set to {start_node}."
                        elif end_node is None:
                            end_node = node
                            all_paths = dijkstra_all_paths(graph, start_node, end_node)
                            concept_text = f"End node set to {end_node}. {len(all_paths)} shortest paths found."
                        break
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_q:
                    running=False
                elif event.key==pygame.K_t:
                    # Toggle traffic lights
                    for k in traffic_lights:
                        traffic_lights[k] = "RED" if traffic_lights[k]=="GREEN" else "GREEN"
                    concept_text = "Traffic lights toggled!"
                elif event.key==pygame.K_c and all_paths:
                    chosen_path = all_paths[0]
                    vehicles.append(Vehicle(f"Car{spawn_id}","car",chosen_path,coords))
                    spawn_id+=1
                    concept_text=f"Car spawned on path P1."
                elif event.key==pygame.K_a and all_paths:
                    chosen_path = all_paths[0]
                    vehicles.append(Vehicle(f"Amb{spawn_id}","ambulance",chosen_path,coords))
                    spawn_id+=1
                    concept_text=f"Ambulance spawned on path P1."

    pygame.quit()

if __name__=="__main__":
    run()
