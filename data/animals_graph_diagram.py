import json
from collections import defaultdict
from typing import Any

from pyvis.network import Network


def load_data(file_path: str) -> list[dict]:
    """
    Load JSONL data from a file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list[dict]: A list of parsed JSON objects.
    """
    data = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {line}\nError: {e}")
    return data


def remove_single_link_nodes_pyvis(net: Network):
    """
    Remove nodes with only a single link from a Pyvis Network.

    Args:
        net (Network): The Pyvis Network object.
    """
    # Count edges for each node
    node_links = defaultdict(int)
    for edge in net.edges:
        node_links[edge["from"]] += 1
        node_links[edge["to"]] += 1

    # Identify nodes with only one link
    single_link_nodes = {node for node, count in node_links.items() if count == 1}

    # Filter nodes and edges
    net.nodes = [node for node in net.nodes if node["id"] not in single_link_nodes]
    net.edges = [
        edge
        for edge in net.edges
        if edge["from"] not in single_link_nodes and edge["to"] not in single_link_nodes
    ]


def create_pyvis_graph(data: list[dict], output_file: str):
    """
    Create an interactive Pyvis graph linking animals based on shared metadata values.

    Args:
        data (list[dict]): List of data items with `id` and `metadata`.
        output_file (str): Path to save the HTML file with the interactive graph.
    """
    # Initialize Pyvis Network
    net = Network(height="800px", width="100%", notebook=False, cdn_resources="remote")
    net.toggle_physics(True)  # Enable physics for node dragging

    # Collect metadata relationships
    for animal in data:
        animal_id = animal["id"]
        metadata: dict[str, Any] = animal.get("metadata", {})
        net.add_node(animal_id, label=animal_id, color="lightblue", physics=True)

        for key, value in metadata.items():
            if key in ["weight", "tail_length", "number_of_legs"]:
                continue
            if key == "keywords":
                for item in value:
                    node_id = f"keyword_{item}"
                    if node_id not in net.node_map:
                        net.add_node(node_id, label=item, color="yellow", physics=True)
                    net.add_edge(animal_id, node_id, title="keyword")
            else:
                node_id = f"{key}_{value}"
                if key == "habitat":
                    color = "lightgreen"
                elif key == "type":
                    color = "lightpink"
                elif key == "diet":
                    color = "orange"
                else:
                    color = "lightpurple"

                if node_id not in net.node_map:
                    net.add_node(node_id, label=value, color=color, physics=True)
                net.add_edge(animal_id, node_id, title=key)

    remove_single_link_nodes_pyvis(net=net)

    # Generate and save the interactive graph
    net.write_html(output_file)

    # Add custom JavaScript for toggling physics only for animal nodes
    with open(output_file, "r") as f:
        html = f.read()

    toggle_script = """
    <script type="text/javascript">
        function togglePhysics(color) {
            // Access the nodes from network.body.data.nodes
            network.body.data.nodes.update(
                network.body.data.nodes.get().map(node => {
                    if (node.color === color) { // Target animal nodes by color
                        node.physics = !node.physics; // Toggle the physics property
                    }
                    return node; // Update the node
                })
            );
        }
    </script>
    <button onclick="togglePhysics('lightblue')" style="background-color: lightblue;">Toggle Animal Physics</button>
    <button onclick="togglePhysics('yellow')" style="background-color: yellow;">Toggle Keyword Physics</button>
    <button onclick="togglePhysics('lightgreen')" style="background-color: lightgreen;">Toggle Habitat Physics</button>
    <button onclick="togglePhysics('lightpink')" style="background-color: lightpink;">Toggle Type Physics</button>
    <button onclick="togglePhysics('orange')" style="background-color: orange;">Toggle Diet Physics</button>
    """

    # Insert the toggle button above the graph container
    html = html.replace("<body>", f"<body>{toggle_script}")

    # Save the updated HTML
    with open(output_file, "w") as f:
        f.write(html)


if __name__ == "__main__":
    # Path to your JSONL file
    file_path = "tests/data/animals.jsonl"

    # Load the data
    data = load_data(file_path)

    create_pyvis_graph(data=data, output_file="tests/data/animals.html")

    # # Create the graph
    # graph = create_graph(data)

    # # Visualize the graph
    # visualize_graph(graph)
