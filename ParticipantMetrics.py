import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean, mode
import randomcolor
import os

import osmnx as ox
import networkx as nx
import folium
from mappymatch.constructs.geofence import Geofence
from mappymatch.constructs.trace import Trace
from geopy import distance

from sklearn import cluster
import hdbscan


class ParticipantMetrics:
    def __init__(self, all_trips):
        """
        constructor for participant level metrics

        :param all_trips: (pd dataframe) trips for a participant
        """
        self.uid = all_trips.uid.iloc[0]
        self.trips = all_trips
        self.home = None
        self.graph = None

        self._centroids = None

    def get_home_location(
        self, radius=5 / 6371, method="MeanShift", adjust_home_index=True
    ):
        """
        Uses clustering to predict a home location from the most common trip start and end points, will only cluster if
        clustering has not already been done using origin_destination clustering. Sets home cluster label to -1
        :param radius: in degrees, works as epsilon for DBSCAN and HDBSCAN, or bandwidth for Mean Shift clustering
        :param method: clustering method (meanshift, dbscan, or hdbscan)
        :param adjust_home_index: bool, will set the cluster label corresponding to the home location to -1 if true
        :return: home location in (lat,lon)
        """
        try:
            getattr(self.trips, "origin_id")
            getattr(self.trips, "destination_id")
        except AttributeError:
            self.origin_destination_clustering(radius, method)

        # identify home cluster
        max_cluster = self.biggest_cluster(
            self.trips.origin_id.to_list() + self.trips.destination_id.to_list()
        )

        # identify home point
        self.home = (self._centroids[max_cluster][1], self._centroids[max_cluster][0])

        # adjust label for home cluster
        if adjust_home_index:
            self.trips.origin_id.loc[self.trips.origin_id == max_cluster] = -1
            self.trips.destination_id.loc[self.trips.destination_id == max_cluster] = -1

        return self.home, self.trips.origin_id, self.trips.destination_id

    def origin_destination_clustering(
        self,
        radius=5 / 6371,
        method="MeanShift",
    ):
        """
        Uses clustering to assign a unique identifier to each destination
        :param radius: float in degrees, works as epsilon for DBSCAN and HDBSCAN, or bandwidth for Mean Shift clustering
        :param method: string clustering method (meanshift, dbscan, or hdbscan)
        :return: list of labels identifying destination clusters
        """
        # check there are the same number of origins and destinations
        len_set = {
            len(self.trips.TSLat),
            len(self.trips.TELat),
            len(self.trips.TSLong),
            len(self.trips.TELong),
        }
        if len(len_set) != 1:
            raise IndexError(
                "The number of origins and destination latitudes and longitudes do not match, check data"
            )
        else:
            # combine lat and lon of origins and  destinations
            y = self.trips.TSLat.to_list() + self.trips.TELat.to_list()
            x = self.trips.TSLong.to_list() + self.trips.TELong.to_list()

            # reshape
            X = np.array([x, y]).transpose()

            # cluster
            if method.lower() == "meanshift":
                labels, self._centroids = self.mean_shift_cluster(X, radius)
            elif method.lower() == "dbscan":
                labels, self._centroids = self.dbscan_cluster(X, radius)
            elif method.lower() == "hdbscan":
                labels, self._centroids = self.hdbscan_cluster(X, radius)
            else:
                raise NotImplementedError("Clustering method not implemented")

        self.trips["origin_id"] = labels[: int(len(labels) / 2)]
        self.trips["destination_id"] = labels[int(len(labels) / 2) :]

        # identify trips based on start and end points
        self.trips["trip_id"] = self.trips.groupby(
            ["origin_id", "destination_id"]
        ).ngroup()

        # get cumulative count of destination
        self.trips["cumulative_trip_count"] = (
            self.trips.groupby("trip_id").cumcount() + 1
        )

        # get total trip count
        self.trips["total_trip_count"] = self.trips.trip_id.groupby(
            self.trips.trip_id
        ).transform("count")

        return (
            self.trips.origin_id,
            self.trips.destination_id,
            self.trips.trip_id,
            self.trips.cumulative_trip_count,
            self.trips.total_trip_count,
        )

    def mean_shift_cluster(self, X, allowable_radius):
        """

        @param X: data to be clustered
        @param allowable_radius: bandwidth for MeanShuft algorithm
        @return: cluster labels for each point and centroids for each label
        """
        ms = cluster.MeanShift(
            bandwidth=allowable_radius, min_bin_freq=5, cluster_all=True
        )
        labels = ms.fit_predict(X)
        centroids = ms.cluster_centers_
        return labels, centroids

    def dbscan_cluster(self, X, allowable_radius):
        """

        @param X: data to be clustered
        @param allowable_radius: epsilon parameter for DBSCAN
        @return: cluster labels for each point and centroids for each label
        """
        dbs = cluster.DBSCAN(eps=allowable_radius, min_samples=5)
        labels = dbs.fit_predict(X)
        centroids = self.get_centroid(X[:, 1], X[:, 0], labels)
        return labels, centroids

    def hdbscan_cluster(self, X, radius):
        """

        @param X: data to be clustered
        @param radius: cluster selection epsilon for HDBSCAN
        @return: cluster labels for each point and centroids for each label
        """
        hdbs = hdbscan.HDBSCAN(min_samples=5, cluster_selection_epsilon=radius)
        labels = hdbs.fit_predict(X)
        centroids = self.get_centroid(X[:, 1], X[:, 0], labels)
        return labels, centroids

    def get_centroid(self, latitude: list, longitude: list, labels: list):
        """
        Finds the centroids as the mean lat lon of a cluster
        @param latitude:  list of latitudes
        @param longitude: list of longitudes
        @param labels: list of clustering labels
        @return: Centroid as ndarray with [lat_i,lon_i]
        """
        data = pd.DataFrame({"lat": latitude, "lon": longitude, "cluster": labels})
        centroid = np.ndarray(shape=(len(data.cluster.unique()), 2))
        for i, c in enumerate(data.cluster.unique()):
            temp = data.loc[data.cluster == c]
            centroid[i][0] = mean(temp.lon)
            centroid[i][1] = mean(temp.lat)

        return centroid

    def biggest_cluster(self, labels: list) -> int:
        """
        Identifies the largest cluster by determining which label is the most common
        @param labels: labels returned from clustering
        @return: label corresponding to the largest cluster
        """
        label, counts = np.unique(labels, return_counts=True)
        count_dict = dict(zip(label, counts))
        count_dict = sorted(count_dict.items(), key=lambda x: x[1])
        if count_dict[-1][0] == -1:
            return count_dict[-2][0]
        else:
            return count_dict[-1][0]

    def map_clusters(
        self,
        style="OpenStreetMap",
        hue="blue",
        home_color="black",
        plot_home: bool = True,
    ):
        """
        @param home_color: colour for home cluster
        @param style: style of folium base map
        @param hue: hue for clusters
        @param plot_home: Whether to plot home marker or not
        @return:
        """
        if self.home:
            location = self.home
        else:
            location = [mean(self.trips.TSLat), mean(self.trips.TSLong)]

        home_map = folium.Map(
            location=location,
            zoom_start=15,
            tiles=style,
        )

        all_labels = (
            self.trips.origin_id.to_list() + self.trips.destination_id.to_list()
        )
        col = {int(l): randomcolor.RandomColor().generate(hue=hue) for l in all_labels}

        for i, (lat, lon, c) in enumerate(
            zip(
                self.trips.TSLat.to_list() + self.trips.TELat.to_list(),
                self.trips.TSLong.to_list() + self.trips.TELong.to_list(),
                all_labels,
            )
        ):
            if c == -1:
                current_col = home_color
            else:
                current_col = col[c]

            # if start point
            if i < len(all_labels) / 2:
                folium.CircleMarker(
                    location=(lat, lon),
                    color=current_col,
                    fill_color=None,
                    radius=5,
                    weight=2,
                ).add_to(home_map)
            # if end point
            else:
                folium.CircleMarker(
                    location=(lat, lon),
                    color=current_col,
                    fill_color=current_col,
                    fill_opacity=0.6,
                    radius=5,
                    weight=2,
                ).add_to(home_map)

        if plot_home and self.home:
            folium.Marker(
                location=self.home,
                icon=folium.Icon(
                    color="red",
                    icon="home",
                    prefix="fa",
                    size=(20, 20),
                ),
            ).add_to(home_map)

        return home_map

    def in_bbox(self, lat_list, lon_list, N, S, E, W) -> bool:
        """
        checks if a trip is entirely are within a box bounded by N,S,E and W
        @param lat_list: list of latitudes in trip
        @param lon_list: list of longitudes in trip
        @param N: max latitude
        @param S: min latitude
        @param E: max longitude
        @param W: min longitude
        @return: True if entire trip is in bbox
        """
        return all(N > lat > S for lat in lat_list) and all(
            E > lon > W for lon in lon_list
        )

    def get_graph(self, buffer=0.3, radius=5 / 6371, method="MeanShift"):
        """
        Gets OSMNx graph for participant (starts with bbox around home location and gets graph with buffer of trips outside the bbox)
        Only computes home location if not already done
        @param buffer: amount of graph to get surrounding trip trajectory in degrees
        @param radius: in degrees, works as epsilon for DBSCAN and HDBSCAN, or bandwidth for Mean Shift clustering for home location
        @param method:  clustering method to get home location (meanshift, dbscan, or hdbscan)
        @return:graph, list of indexes where graph could not be obtained
        """

        if self.home is None:
            self.get_home_location(radius, method)

        (N, S, E, W) = (
            self.home[0] + buffer,
            self.home[0] - buffer,
            self.home[1] + buffer,
            self.home[1] - buffer,
        )
        # get initial graph around home location
        G = ox.graph_from_bbox(
            N, S, E, W, network_type="drive", truncate_by_edge=True, simplify=False
        )
        no_graph = []
        count = 0

        for idx, trip in self.trips.iterrows():
            count += 1
            if self.in_bbox(trip.Latitude, trip.Longitude, N, S, E, W):
                # clear_output(wait=True)
                print(
                    f"UID {self.uid}: Completed for {count}/{len(self.trips)} ({round(count / len(self.trips), 2) * 100}"
                    f"%) trips, {len(no_graph)} trips with no graph",
                    end="\r",
                    flush=True,
                )
                continue
            else:
                temp = pd.DataFrame(
                    {"latitude": trip.Latitude, "longitude": trip.Longitude}
                )
                trace = Trace.from_dataframe(temp)
                geofence = Geofence.from_trace(trace, padding=1e3)
                try:
                    H = ox.graph_from_polygon(
                        geofence.geometry,
                        network_type="drive",
                        truncate_by_edge=True,
                    )
                    G = nx.compose(G, H)
                except:
                    no_graph.append(idx)
                    continue

        # clear_output(wait=True)
        print(
            f"UID {self.uid}: Completed for {count}/{len(self.trips)} ({round(count / len(self.trips), 2) * 100}%) "
            f"trips, {len(no_graph)} trips with no graph",
            end="\r",
            flush=True,
        )

        self.graph = G
        return G, no_graph

    def save_graph_graphml(self, folderpath=None):
        """
        Saves graph as a graphml file
        @param folderpath: string os folder to save graph
        @return: None
        """
        if folderpath is not None:
            filepath = f"{folderpath}/{self.uid}"
        else:
            filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

        if self.graph is None:
            print(
                f"Graph non-existent for participant {self.uid}... obtaining graph with default parameters"
            )
            self.get_graph()
        ox.io.save_graphml(self.graph, filepath=filepath)

    def read_graph_graphml(self, filepath=None):
        """
        Reads graph from a graphml file
        @param filepath: string os path
        @return: ox graph
        """
        try:
            self.graph = ox.load_graphml(filepath)
        except ValueError:
            pass

        try:
            self.graph = ox.load_graphml(filepath=f"{filepath}/{self.uid}")
        except ValueError:
            print("Could not  load graph, check filepath")

        return self.graph

    def save_graph_shapefile_directional(self, G, folderpath=None, encoding="utf-8"):
        """
        https://github.com/cyang-kth/osm_mapmatching

        @param G: ox graph
        @param folderpath: string os folder to save graph
        @param encoding: ???
        @return: None
        """
        # default filepath if no filename was provided
        if folderpath is not None:
            filepath = f"{folderpath}/{self.uid}"
        else:
            filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

        # if save folder does not already exist, create it (shapefiles
        # get saved as set of files)
        if not filepath == "" and not os.path.exists(filepath):
            os.makedirs(filepath)
        filepath_nodes = os.path.join(filepath, "nodes.shp")
        filepath_edges = os.path.join(filepath, "edges.shp")

        # convert undirected graph to gdfs and stringify non-numeric columns
        gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
        gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
        gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
        # We need an unique ID for each edge
        gdf_edges["fid"] = np.arange(0, gdf_edges.shape[0], dtype="int")
        # save the nodes and edges as separate ESRI shapefiles
        gdf_nodes.to_file(filepath_nodes, encoding=encoding)
        gdf_edges.to_file(filepath_edges, encoding=encoding)
