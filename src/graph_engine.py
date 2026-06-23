import osmnx as ox
import networkx as nx
import os
import pickle
import itertools
import numpy as np
import pandas as pd
import pyproj
import re

# NOTE: torch and scipy.sparse imports removed — compute_scaled_laplacian was dead
# code that was never called anywhere in the pipeline. Removing it also removes
# the 2GB torch dependency from requirements.txt.


class BengaluruSpatialGraph:
    def __init__(self, cache_dir="data"):
        self.cache_path = os.path.join(cache_dir, "bengaluru_drive_graph.pkl")
        self.G = None
        self._load_or_download_graph()
        self.transformer = pyproj.Transformer.from_crs(
            "epsg:4326", "epsg:32643", always_xy=True
        )
        # Inverse transform: projected UTM (x,y) -> geographic (lon,lat).
        # Needed because after ox.project_graph(), node 'x'/'y' hold metres,
        # not lat/lon — used by get_diversion_routes() and the approach-road
        # bearing fix below.
        self.inv_transformer = pyproj.Transformer.from_crs(
            "epsg:32643", "epsg:4326", always_xy=True
        )

    def _load_or_download_graph(self):
        # Cache locally so we don't hammer the Overpass API on every run
        if os.path.exists(self.cache_path):
            print("Loading Bengaluru road graph from local cache...")
            with open(self.cache_path, "rb") as f:
                self.G = pickle.load(f)
        else:
            print("Downloading Bengaluru drivable network from OpenStreetMap...")
            self.G = ox.graph_from_place(
                "Bengaluru, Karnataka, India", network_type="drive"
            )
            # Project to UTM zone 43N for accurate metric distance calculations
            self.G = ox.project_graph(self.G)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.G, f)
            print("Graph cached.")

    def snap_coordinates_to_network(self, lat: float, lon: float) -> dict:
        """Snap raw GPS coordinates to the nearest road segment edge in the OSM graph."""
        try:
            if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
                return self._empty_edge_metrics()

            # Graph is projected in UTM 43N — transform input coords first
            x, y = self.transformer.transform(lon, lat)

            u, v, key = ox.distance.nearest_edges(self.G, X=x, Y=y)
            edge_data = self.G[u][v][key]

            road_name = str(edge_data.get("name", "Unknown Road"))

            corridor = "OTHER_CBD_ROAD"
            if any(
                kw in road_name.lower()
                for kw in ["outer ring road", "orr", "marathahalli", "silk board", "hebbal"]
            ):
                corridor = "OUTER_RING_ROAD"
            elif any(
                kw in road_name.lower()
                for kw in ["tumkur road", "hosur road", "bellary road", "mysore road"]
            ):
                corridor = "PRIMARY_NATIONAL_HIGHWAY"

            is_bridge = 1.0 if (
                "flyover" in road_name.lower() or "bridge" in road_name.lower()
            ) else 0.0

            lanes_raw = edge_data.get("lanes", 1)
            if isinstance(lanes_raw, list):
                lanes = int(lanes_raw[0]) if lanes_raw else 1
            elif isinstance(lanes_raw, str):
                try:
                    lanes = int(lanes_raw)
                except ValueError:
                    lanes = 1
            elif isinstance(lanes_raw, (int, float)):
                lanes = int(lanes_raw)
            else:
                lanes = 1

            maxspeed_raw = edge_data.get("maxspeed", 40)
            if isinstance(maxspeed_raw, list):
                maxspeed_raw = maxspeed_raw[0]
            if isinstance(maxspeed_raw, str):
                digits = re.findall(r"\d+", maxspeed_raw)
                maxspeed = float(digits[0]) if digits else 40.0
            elif isinstance(maxspeed_raw, (int, float)):
                maxspeed = float(maxspeed_raw)
            else:
                maxspeed = 40.0

            return {
                "osm_edge_u": u,
                "osm_edge_v": v,
                "road_name": road_name,
                "road_highway_type": str(edge_data.get("highway", "unclassified")),
                "road_lanes": lanes,
                "road_maxspeed": maxspeed,
                "corridor_association": corridor,
                "bottleneck_proximity_index": float(0.85 if is_bridge else 0.32),
            }
        except Exception:
            return self._empty_edge_metrics()

    def snap_coordinates_to_network_vectorized(self, lats, lons) -> list:
        """Batch coordinate snapping — much faster than calling snap one-by-one."""
        valid_mask = ~(pd.isna(lats) | pd.isna(lons) | (lats == 0) | (lons == 0))
        results = [self._empty_edge_metrics() for _ in range(len(lats))]

        if not valid_mask.any():
            return results

        valid_indices = np.where(valid_mask)[0]
        valid_lats = np.array(lats)[valid_mask]
        valid_lons = np.array(lons)[valid_mask]

        try:
            x, y = self.transformer.transform(valid_lons, valid_lats)
            edges = ox.distance.nearest_edges(self.G, X=x, Y=y)

            for idx, (u, v, key) in zip(valid_indices, edges):
                try:
                    edge_data = self.G[u][v][key]
                    road_name = str(edge_data.get("name", "Unknown Road"))

                    corridor = "OTHER_CBD_ROAD"
                    if any(
                        kw in road_name.lower()
                        for kw in ["outer ring road", "orr", "marathahalli", "silk board", "hebbal"]
                    ):
                        corridor = "OUTER_RING_ROAD"
                    elif any(
                        kw in road_name.lower()
                        for kw in ["tumkur road", "hosur road", "bellary road", "mysore road"]
                    ):
                        corridor = "PRIMARY_NATIONAL_HIGHWAY"

                    is_bridge = 1.0 if (
                        "flyover" in road_name.lower() or "bridge" in road_name.lower()
                    ) else 0.0

                    lanes_raw = edge_data.get("lanes", 1)
                    if isinstance(lanes_raw, list):
                        lanes = int(lanes_raw[0]) if lanes_raw else 1
                    elif isinstance(lanes_raw, str):
                        try:
                            lanes = int(lanes_raw)
                        except ValueError:
                            lanes = 1
                    elif isinstance(lanes_raw, (int, float)):
                        lanes = int(lanes_raw)
                    else:
                        lanes = 1

                    maxspeed_raw = edge_data.get("maxspeed", 40)
                    if isinstance(maxspeed_raw, list):
                        maxspeed_raw = maxspeed_raw[0]
                    if isinstance(maxspeed_raw, str):
                        digits = re.findall(r"\d+", maxspeed_raw)
                        maxspeed = float(digits[0]) if digits else 40.0
                    elif isinstance(maxspeed_raw, (int, float)):
                        maxspeed = float(maxspeed_raw)
                    else:
                        maxspeed = 40.0

                    results[idx] = {
                        "osm_edge_u": u,
                        "osm_edge_v": v,
                        "road_name": road_name,
                        "road_highway_type": str(edge_data.get("highway", "unclassified")),
                        "road_lanes": lanes,
                        "road_maxspeed": maxspeed,
                        "corridor_association": corridor,
                        "bottleneck_proximity_index": float(0.85 if is_bridge else 0.32),
                    }
                except Exception:
                    pass
        except Exception:
            pass

        return results

    def get_approach_roads(
        self, lat: float, lon: float, radius_m: int = 350, max_approaches: int = 4
    ) -> list:
        """
        For any event GPS point, find the real approach roads within radius_m metres
        using an ego-graph traversal on the loaded OSM network.

        Returns a list of approach dicts with real road names, OSM node IDs,
        bearing estimates, and GPS offsets — exactly what barricade_engine needs
        to place barricades on actual roads instead of cardinal-direction guesses.

        Used by barricade_engine.generate_barricade_plan() as the primary data
        source when junction_name is not in the hardcoded junction table.
        """
        try:
            if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
                return []

            # Step 1: snap event point to nearest OSM node
            x_event, y_event = self.transformer.transform(lon, lat)
            nearest_node = ox.distance.nearest_nodes(self.G, X=x_event, Y=y_event)

            # Step 2: build ego-graph — all nodes within radius_m of event node
            ego = nx.ego_graph(self.G, nearest_node, radius=radius_m, distance="length")

            approaches = []
            seen_roads = set()

            for u, v, data in ego.edges(data=True):
                # Identify the "far" endpoint of this approach road relative to
                # the event node. Edges exist in both directions (u->v and
                # v->u) in OSM's directed graph, so we must check BOTH ends —
                # checking only u==nearest_node (as a previous version did)
                # let the reverse-direction edge slip through and incorrectly
                # treat the event's own node as the far end, zeroing the offset
                # for roughly half of all approach roads found.
                if u == nearest_node:
                    other_node = v
                elif v == nearest_node:
                    other_node = u
                else:
                    # Edge between two non-event nodes further out in the
                    # ego-graph — not a direct approach from the event point.
                    continue
                if other_node == nearest_node:
                    continue  # true self-loop

                road_name = str(data.get("name", "Unknown Road"))
                if isinstance(road_name, list):
                    road_name = road_name[0]

                # De-duplicate by road name — we only need one approach per road
                road_key = road_name.lower().strip()
                if road_key in seen_roads or road_key == "unknown road":
                    continue
                seen_roads.add(road_key)

                # Get the far node's position to compute bearing and GPS offset.
                # NOTE: after ox.project_graph(), node 'x'/'y' are projected UTM
                # metres, not lat/lon — inverse-transform them back to geographic
                # coordinates. (Previous version read non-existent 'lat'/'lon'
                # keys here, which silently collapsed every offset to zero.)
                node_data = self.G.nodes[other_node]
                node_x, node_y = node_data.get("x"), node_data.get("y")
                if node_x is not None and node_y is not None:
                    node_lon, node_lat = self.inv_transformer.transform(node_x, node_y)
                else:
                    node_lat, node_lon = lat, lon

                # Estimate bearing: angle from event point to this approach road node
                dlat = node_lat - lat
                dlon = node_lon - lon
                import math
                bearing = math.degrees(math.atan2(dlon, dlat)) % 360

                # Offset coordinates: place barricade 200m back from event on this road
                lat_offset = dlat * 0.6
                lon_offset = dlon * 0.6

                highway_type = str(data.get("highway", "unclassified"))
                if isinstance(highway_type, list):
                    highway_type = highway_type[0]

                # Priority: primary/trunk roads get barricaded first
                priority = "HIGH" if highway_type in (
                    "primary", "trunk", "secondary", "motorway"
                ) else "MEDIUM"

                approaches.append({
                    "name": road_name,
                    "bearing": round(bearing, 1),
                    "lat_offset": round(lat_offset, 6),
                    "lng_offset": round(lon_offset, 6),
                    "highway_type": highway_type,
                    "osm_node_u": nearest_node,
                    "osm_node_v": other_node,
                    "priority": priority,
                })

                if len(approaches) >= max_approaches:
                    break

            return approaches

        except Exception as e:
            print(f"  get_approach_roads failed ({e}) — barricade_engine will use cardinal fallback")
            return []

    def get_diversion_routes(
        self,
        event_lat: float,
        event_lon: float,
        num_routes: int = 2,
        block_radius_m: int = 300,
        anchor_radius_m: int = 900,
    ) -> list:
        """
        Compute real OSM shortest-path diversion routes that AVOID the blocked
        event zone — answers "where do diverted vehicles go?" for the map UI.

        How it works:
          1. Snap the event point to the nearest OSM node.
          2. Carve out a local subgraph within anchor_radius_m * 2.2 of the
             event (keeps the search fast regardless of the full city graph size).
          3. Remove every node within block_radius_m of the event (the blocked
             zone) from a routable copy of that subgraph.
          4. Pick two boundary nodes on roughly opposite sides of the block —
             these represent "traffic arriving from one side" / "traffic that
             needs to reach the other side".
          5. Compute the shortest path between them on the routable graph
             (which physically cannot pass through the blocked zone).
          6. To get a second, genuinely different alternate route, heavily
             penalize the edges used by route 1 and recompute the shortest
             path — a standard, MultiDiGraph-safe way to approximate k-shortest
             paths (nx.shortest_simple_paths/Yen's algorithm does not support
             OSMnx's MultiDiGraph, so we don't use it here).

        Returns a list of dicts (best/shortest first), each with:
          coords      : list of (lat, lon) tuples — ready for folium.PolyLine
          length_km   : route length in km
          eta_min     : rough ETA at a diverted-traffic speed (~22 km/h)
          via_roads   : up to 5 real road names traversed (deduplicated, in order)
          n_nodes     : number of OSM nodes in the path (diagnostic only)

        Returns [] if no route could be found (e.g. coordinate outside the
        cached network, or the blocked zone fully disconnects the local area) —
        callers should treat that as "show barricades only, no diversion path".
        """
        try:
            if pd.isna(event_lat) or pd.isna(event_lon) or event_lat == 0 or event_lon == 0:
                return []

            x_ev, y_ev = self.transformer.transform(event_lon, event_lat)
            event_node = ox.distance.nearest_nodes(self.G, X=x_ev, Y=y_ev)

            # Step 1: local subgraph — bounds the computation regardless of
            # how large the full Bengaluru drive network is.
            search_radius_m = anchor_radius_m * 2.2
            local = nx.ego_graph(self.G, event_node, radius=search_radius_m, distance="length").copy()

            # Step 2: carve out the blocked zone
            blocked_ego = nx.ego_graph(self.G, event_node, radius=block_radius_m, distance="length")
            blocked_nodes = set(blocked_ego.nodes()) & set(local.nodes())

            routable = local.copy()
            routable.remove_nodes_from(blocked_nodes)

            # Step 3: boundary candidates — nodes just outside the block,
            # still inside the local subgraph
            anchor_ego = nx.ego_graph(self.G, event_node, radius=anchor_radius_m, distance="length")
            boundary_candidates = [n for n in anchor_ego.nodes() if n in routable.nodes()]

            if len(boundary_candidates) < 2:
                return []

            # Step 4: pick the two boundary nodes farthest apart (≈ opposite
            # sides of the blocked junction) — capped sample for speed
            sample = boundary_candidates if len(boundary_candidates) <= 50 else \
                list(np.random.RandomState(7).choice(boundary_candidates, 50, replace=False))

            best_pair, best_d2 = None, -1.0
            for a, b in itertools.combinations(sample, 2):
                ax, ay = routable.nodes[a]["x"], routable.nodes[a]["y"]
                bx, by = routable.nodes[b]["x"], routable.nodes[b]["y"]
                d2 = (ax - bx) ** 2 + (ay - by) ** 2
                if d2 > best_d2:
                    best_d2, best_pair = d2, (a, b)

            if best_pair is None:
                return []
            origin_node, dest_node = best_pair

            # Step 5 & 6: primary shortest path, then penalize-and-reroute for
            # additional alternates (works natively on MultiDiGraph, unlike
            # nx.shortest_simple_paths / Yen's algorithm)
            work_graph = routable.copy()
            raw_paths = []
            for _ in range(max(num_routes, 1)):
                try:
                    path = nx.shortest_path(work_graph, origin_node, dest_node, weight="length")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    break
                raw_paths.append(path)
                for u, v in zip(path[:-1], path[1:]):
                    if work_graph.has_edge(u, v):
                        for k in list(work_graph[u][v].keys()):
                            cur = work_graph[u][v][k].get("length", 1.0) or 1.0
                            work_graph[u][v][k]["length"] = cur * 6.0

            if not raw_paths:
                return []

            # Convert node paths -> lat/lon polylines + real road names
            results = []
            for path in raw_paths:
                coords, road_names, length_m = [], [], 0.0
                for n in path:
                    x, y = self.G.nodes[n]["x"], self.G.nodes[n]["y"]
                    lon, lat = self.inv_transformer.transform(x, y)
                    coords.append((lat, lon))
                for u, v in zip(path[:-1], path[1:]):
                    edge_data = self.G.get_edge_data(u, v)
                    if not edge_data:
                        continue
                    ed = edge_data[next(iter(edge_data))]
                    length_m += float(ed.get("length", 0) or 0)
                    nm = ed.get("name")
                    if isinstance(nm, list):
                        nm = nm[0] if nm else None
                    if nm and nm not in road_names:
                        road_names.append(str(nm))

                length_km = round(length_m / 1000, 2)
                results.append({
                    "coords": coords,
                    "length_km": length_km,
                    "eta_min": round(length_km / 22.0 * 60, 1) if length_km else 0.0,
                    "via_roads": road_names[:5] if road_names else ["Unnamed local road"],
                    "n_nodes": len(path),
                })

            # De-dup near-identical routes (penalize-and-reroute can sometimes
            # converge back onto the same physical road if there's only one
            # viable corridor)
            seen, deduped = set(), []
            for r in results:
                sig = (r["length_km"], tuple(r["via_roads"]))
                if sig in seen:
                    continue
                seen.add(sig)
                deduped.append(r)

            deduped.sort(key=lambda r: r["length_km"])
            return deduped[:num_routes]

        except Exception as e:
            print(f"  get_diversion_routes failed ({e}) — barricade-only plan will be shown")
            return []

    def _empty_edge_metrics(self):
        return {
            "osm_edge_u": -1,
            "osm_edge_v": -1,
            "road_highway_type": "UNKNOWN",
            "road_lanes": 1,
            "road_maxspeed": 40.0,
        }
