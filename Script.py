import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import json
from pathlib import Path
from datetime import datetime
import traceback

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import (
    GCNConv, GATConv, RGCNConv,
    global_mean_pool, global_max_pool, global_add_pool
)


# ==========================
# Model Definitions
# ==========================
class EnhancedGCNModel(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_heads=4, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_features + 1, hidden_dim, padding_idx=0)

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self._dropout = dropout

    def forward(self, x, edge_index, batch=None):
        dropout = self._dropout

        x = torch.clamp(x, 0, self.embedding.num_embeddings - 1)
        if x.dim() == 1:
            x = self.embedding(x)

        # Ensure a valid edge_index
        if edge_index.numel() == 0 or edge_index.size(0) != 2:
            num_nodes = x.size(0)
            edge_index = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)

        max_node_idx = x.size(0) - 1
        edge_index = torch.clamp(edge_index, 0, max_node_idx)

        try:
            x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
            x1 = F.dropout(x1, training=self.training, p=dropout)

            x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
            x2 = F.dropout(x2, training=self.training, p=dropout)
            x2 = x2 + x1

            x3 = F.relu(self.bn3(self.conv3(x2, edge_index)))
            x3 = x3 + x2

            if x3.size(0) > 1:
                x_att, _ = self.attention(x3.unsqueeze(0), x3.unsqueeze(0), x3.unsqueeze(0))
                x_final = x3 + x_att.squeeze(0)
            else:
                x_final = x3
        except (RuntimeError, ValueError):
            x_final = x.mean(dim=0, keepdim=True).expand(x.size(0), -1)

        if batch is None:
            batch = torch.zeros(x_final.size(0), dtype=torch.long, device=x_final.device)

        batch = torch.clamp(batch, 0, x_final.size(0) - 1)

        x_mean = global_mean_pool(x_final, batch)
        x_max = global_max_pool(x_final, batch)
        x_sum = global_add_pool(x_final, batch)
        x_combined = torch.cat([x_mean, x_max, x_sum], dim=1)
        return self.classifier(x_combined)


class GATModel(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_heads=4, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_features + 1, hidden_dim, padding_idx=0)

        self.gat1 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, concat=False)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self._dropout = dropout

    def forward(self, x, edge_index, batch=None):
        dropout = self._dropout

        x = torch.clamp(x, 0, self.embedding.num_embeddings - 1)
        if x.dim() == 1:
            x = self.embedding(x)

        if edge_index.numel() == 0 or edge_index.size(0) != 2:
            num_nodes = x.size(0)
            edge_index = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)

        max_node_idx = x.size(0) - 1
        edge_index = torch.clamp(edge_index, 0, max_node_idx)

        try:
            x1 = F.elu(self.gat1(x, edge_index))
            x1 = self.ln1(x1)
            x1 = F.dropout(x1, training=self.training, p=dropout)

            x2 = F.elu(self.gat2(x1, edge_index))
            x2 = self.ln2(x2)
            x2 = F.dropout(x2, training=self.training, p=dropout)
            x2 = x2 + x1

            x3 = F.elu(self.gat3(x2, edge_index))
            x3 = self.ln3(x3)
            x_final = x3
        except (RuntimeError, AssertionError, ValueError):
            x_final = x.mean(dim=0, keepdim=True).expand(x.size(0), -1)

        if batch is None:
            batch = torch.zeros(x_final.size(0), dtype=torch.long, device=x_final.device)

        batch = torch.clamp(batch, 0, x_final.size(0) - 1)
        x_pooled = global_mean_pool(x_final, batch)
        return self.classifier(x_pooled)


class RGCNModel(nn.Module):
    def __init__(self, num_features, num_relations, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.num_relations = max(1, num_relations)
        self.embedding = nn.Embedding(num_features + 1, hidden_dim, padding_idx=0)

        self.rgcn1 = RGCNConv(hidden_dim, hidden_dim, self.num_relations)
        self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, self.num_relations)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self._dropout = dropout

    def forward(self, x, edge_index, edge_type=None, batch=None):
        dropout = self._dropout

        x = torch.clamp(x, 0, self.embedding.num_embeddings - 1)
        if x.dim() == 1:
            x = self.embedding(x)

        if edge_index.numel() == 0 or edge_index.size(0) != 2:
            num_nodes = x.size(0)
            edge_index = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)

        max_node_idx = x.size(0) - 1
        edge_index = torch.clamp(edge_index, 0, max_node_idx)

        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        edge_type = torch.clamp(edge_type, 0, self.num_relations - 1)

        try:
            x1 = F.relu(self.bn1(self.rgcn1(x, edge_index, edge_type)))
            x1 = F.dropout(x1, training=self.training, p=dropout)

            x2 = F.relu(self.bn2(self.rgcn2(x1, edge_index, edge_type)))
            x2 = F.dropout(x2, training=self.training, p=dropout)
            x2 = x2 + x1

            x_final = x2
        except (RuntimeError, ValueError):
            x_final = x.mean(dim=0, keepdim=True).expand(x.size(0), -1)

        if batch is None:
            batch = torch.zeros(x_final.size(0), dtype=torch.long, device=x_final.device)

        batch = torch.clamp(batch, 0, x_final.size(0) - 1)
        x_pooled = global_mean_pool(x_final, batch)
        return self.classifier(x_pooled)


# ==========================
# Model Loader
# ==========================
def load_model(model_type, metadata_path, map_location="cpu"):
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    init_args = meta["init_args"]

    if model_type == "enhanced_gcn":
        model = EnhancedGCNModel(
            init_args["num_features"],
            hidden_dim=init_args.get("hidden_dim", 64),
            num_heads=init_args.get("num_heads", 4),
            dropout=init_args.get("dropout", 0.3),
        )
    elif model_type == "gat":
        model = GATModel(
            init_args["num_features"],
            hidden_dim=init_args.get("hidden_dim", 64),
            num_heads=init_args.get("num_heads", 4),
            dropout=init_args.get("dropout", 0.3),
        )
    elif model_type == "rgcn":
        model = RGCNModel(
            init_args["num_features"],
            init_args.get("num_relations", 1),
            hidden_dim=init_args.get("hidden_dim", 64),
            dropout=init_args.get("dropout", 0.3),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    weights_path = meta["weights_path"]
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(os.path.dirname(metadata_path), weights_path)

    state = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state)
    model.eval()
    return model, meta


# ==========================
# App Class
# ==========================
class ToxinPredictionApp:
    def __init__(self, root):
        # --- App state ---
        self.root = root
        self.root.title("🧬 Toxin Prediction System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Runtime state
        self.models = {}              # name -> (model, meta)
        self.models_dir = str(Path(__file__).parent / "models")
        self.graph_data = None
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.relation_to_idx = {}
        self.toxin_nodes = set()
        self.nodes_df = None
        self.edges_df = None
        self.data_loaded = False
        self.last_results = []        # list of dicts for export

        # UI
        self.notebook = None
        self.setup_tab = None
        self.prediction_tab = None
        self.results_tab = None

        self.status_text = None
        self.model_listbox = None
        self.nodes_file_var = tk.StringVar()
        self.edges_file_var = tk.StringVar()

        self.selected_model_var = tk.StringVar()
        self.model_combo = None
        self.input_nodes_var = tk.StringVar()
        self.topk_var = tk.IntVar(value=10)
        self.results_tree = None

        # Build UI
        self.create_widgets()
        self.load_available_models()

    # --------------------------
    # GUI Setup
    # --------------------------
    def create_widgets(self):
        title_frame = tk.Frame(self.root, bg="#f0f0f0")
        title_frame.pack(fill="x", padx=20, pady=10)

        tk.Label(
            title_frame,
            text="🧬 Toxin Prediction System",
            font=("Arial", 20, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50",
        ).pack()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)

        self.setup_tab = tk.Frame(self.notebook, bg="white")
        self.prediction_tab = tk.Frame(self.notebook, bg="white")
        self.results_tab = tk.Frame(self.notebook, bg="white")

        self.notebook.add(self.setup_tab, text="📂 Setup")
        self.notebook.add(self.prediction_tab, text="🔮 Prediction")
        self.notebook.add(self.results_tab, text="📊 Results")

        self.create_setup_tab()
        self.create_prediction_tab()
        self.create_results_tab()

    def create_setup_tab(self):
        model_frame = tk.LabelFrame(
            self.setup_tab, text="Models", font=("Arial", 12, "bold"), bg="white", padx=10, pady=10
        )
        model_frame.pack(fill="x", padx=20, pady=10)

        tk.Button(
            model_frame,
            text="🔄 Refresh Models",
            command=self.load_available_models,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(side="left", padx=5)

        tk.Button(
            model_frame,
            text="📁 Browse Models Folder",
            command=self.browse_models_folder,
            bg="#2ecc71",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(side="left", padx=5)

        self.model_listbox = tk.Listbox(model_frame, height=6, font=("Arial", 10))
        self.model_listbox.pack(fill="x", pady=10)

        data_frame = tk.LabelFrame(
            self.setup_tab, text="Data Files", font=("Arial", 12, "bold"), bg="white", padx=10, pady=10
        )
        data_frame.pack(fill="x", padx=20, pady=10)

        nodes_frame = tk.Frame(data_frame, bg="white")
        nodes_frame.pack(fill="x", pady=5)
        tk.Label(nodes_frame, text="Nodes CSV:", bg="white", font=("Arial", 10)).pack(side="left")
        tk.Entry(nodes_frame, textvariable=self.nodes_file_var, width=50).pack(side="left", padx=10)
        tk.Button(nodes_frame, text="Browse", command=self.browse_nodes_file).pack(side="left")

        edges_frame = tk.Frame(data_frame, bg="white")
        edges_frame.pack(fill="x", pady=5)
        tk.Label(edges_frame, text="Edges CSV:", bg="white", font=("Arial", 10)).pack(side="left")
        tk.Entry(edges_frame, textvariable=self.edges_file_var, width=50).pack(side="left", padx=10)
        tk.Button(edges_frame, text="Browse", command=self.browse_edges_file).pack(side="left")

        tk.Button(
            data_frame,
            text="📊 Load Data Files",
            command=self.load_data_files,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 12, "bold"),
        ).pack(pady=10)

        status_frame = tk.LabelFrame(
            self.setup_tab, text="Status", font=("Arial", 12, "bold"), bg="white", padx=10, pady=10
        )
        status_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.status_text = scrolledtext.ScrolledText(status_frame, height=10, font=("Consolas", 9))
        self.status_text.pack(fill="both", expand=True)

        self.log_message("Welcome to Toxin Prediction System!")
        self.log_message("Please load your models and data files to get started.")

    def create_prediction_tab(self):
        input_frame = tk.LabelFrame(
            self.prediction_tab, text="Input Nodes for Prediction", font=("Arial", 12, "bold"), bg="white", padx=10, pady=10
        )
        input_frame.pack(fill="x", padx=20, pady=10)

        # ✅ Updated label (was "Enter node IDs")
        tk.Label(input_frame, text="Enter node names (comma-separated):", bg="white", font=("Arial", 10)).pack(anchor="w")

        self.input_entry = tk.Entry(input_frame, textvariable=self.input_nodes_var, width=80, font=("Arial", 10))
        self.input_entry.pack(fill="x", pady=5)

        settings_frame = tk.LabelFrame(
            self.prediction_tab, text="Prediction Settings", font=("Arial", 12, "bold"), bg="white", padx=10, pady=10
        )
        settings_frame.pack(fill="x", padx=20, pady=10)

        model_select_frame = tk.Frame(settings_frame, bg="white")
        model_select_frame.pack(fill="x", pady=5)
        tk.Label(model_select_frame, text="Select Model:", bg="white", font=("Arial", 10)).pack(side="left")
        self.model_combo = ttk.Combobox(model_select_frame, textvariable=self.selected_model_var, width=40, state="readonly")
        self.model_combo.pack(side="left", padx=10)

        topk_frame = tk.Frame(settings_frame, bg="white")
        topk_frame.pack(fill="x", pady=5)
        tk.Label(topk_frame, text="Top K Predictions:", bg="white", font=("Arial", 10)).pack(side="left")
        tk.Spinbox(topk_frame, from_=1, to=100, textvariable=self.topk_var, width=10).pack(side="left", padx=10)

        tk.Button(
            settings_frame,
            text="🔮 Predict Toxins",
            command=self.run_prediction,
            bg="#9b59b6",
            fg="white",
            font=("Arial", 14, "bold"),
            height=2,
        ).pack(pady=20)

        examples_frame = tk.LabelFrame(
            self.prediction_tab, text="Quick Examples", font=("Arial", 12, "bold"), bg="white", padx=10, pady=10
        )
        examples_frame.pack(fill="x", padx=20, pady=10)

        # ✅ Updated example text (now shows names instead of IDs)
        example_text = (
            "Examples of node name inputs:\n"
            "• Single node: Histone H2A type 1-A\n"
            "• Multiple nodes: Histone H2A type 1-A,drooling,dry eye,dullness\n"
            "• Mixed types (names and IDs): Histone H2A type 1-A,drooling,dry eye,dullness,Oral\n"
        )
        tk.Label(examples_frame, text=example_text, bg="white", font=("Arial", 9), justify="left").pack(anchor="w")

    def create_results_tab(self):
        results_frame = tk.LabelFrame(
            self.results_tab, text="Prediction Results", font=("Arial", 12, "bold"), bg="white", padx=10, pady=10
        )
        results_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Updated columns - removed Description, added Toxin Name
        columns = ("Rank", "Toxin ID", "Toxin Name", "Confidence", "Type")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=15)

        self.results_tree.heading("Rank", text="Rank")
        self.results_tree.heading("Toxin ID", text="Toxin ID")
        self.results_tree.heading("Toxin Name", text="Toxin Name")
        self.results_tree.heading("Confidence", text="Confidence (%)")
        self.results_tree.heading("Type", text="Node Type")

        self.results_tree.column("Rank", width=60, anchor="center")
        self.results_tree.column("Toxin ID", width=120, anchor="center")
        self.results_tree.column("Toxin Name", width=300, anchor="w")
        self.results_tree.column("Confidence", width=120, anchor="center")
        self.results_tree.column("Type", width=120, anchor="center")

        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        self.results_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        export_frame = tk.Frame(self.results_tab, bg="white")
        export_frame.pack(fill="x", padx=20, pady=10)

        tk.Button(
            export_frame,
            text="💾 Export Results",
            command=self.export_results,
            bg="#34495e",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(side="right")
    # --------------------------
    # Logging helper
    # --------------------------
    def log_message(self, message: str):
        if self.status_text is None:
            print(message)
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    # --------------------------
    # Model management
    # --------------------------
    def browse_models_folder(self):
        path = filedialog.askdirectory(title="Select Models Folder", initialdir=self.models_dir)
        if path:
            self.models_dir = path
            self.load_available_models()

    def _discover_metadata_files(self, base_dir: str):
        base = Path(base_dir)
        if not base.exists():
            return []
        # Heuristic: metadata json files that contain required keys.
        return list(base.rglob("*.json"))

    def load_available_models(self):
        self.models.clear()
        if self.model_listbox:
            self.model_listbox.delete(0, tk.END)
        found = 0

        for meta_path in self._discover_metadata_files(self.models_dir):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                # Minimal validation
                if not {"model_type", "init_args", "weights_path"} <= set(meta.keys()):
                    continue
                model_type = meta["model_type"]
                name = meta.get("name") or f"{model_type} @ {meta_path.name}"
                model, meta_loaded = load_model(model_type, str(meta_path), map_location="cpu")
                self.models[name] = (model, meta_loaded)
                if self.model_listbox:
                    self.model_listbox.insert(tk.END, name)
                found += 1
            except Exception as e:
                self.log_message(f"⚠️ Skipped {meta_path}: {e}")

        if self.model_combo:
            self.model_combo["values"] = list(self.models.keys())
            if self.models and not self.selected_model_var.get():
                self.selected_model_var.set(next(iter(self.models.keys())))

        if found == 0:
            self.log_message(f"ℹ️ No models found in: {self.models_dir}")
        else:
            self.log_message(f"✅ Loaded {found} model(s) from: {self.models_dir}")

    # --------------------------
    # Data loading and graph build (NO NetworkX)
    # --------------------------
    def browse_nodes_file(self):
        path = filedialog.askopenfilename(
            title="Select Nodes CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if path:
            self.nodes_file_var.set(path)

    def browse_edges_file(self):
        path = filedialog.askopenfilename(
            title="Select Edges CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if path:
            self.edges_file_var.set(path)

    def load_data_files(self):
        nodes_path = self.nodes_file_var.get().strip()
        edges_path = self.edges_file_var.get().strip()
        if not nodes_path or not edges_path:
            messagebox.showwarning("Missing Files", "Please select both Nodes and Edges CSV files.")
            return
        try:
            self.nodes_df = pd.read_csv(nodes_path)
            self.edges_df = pd.read_csv(edges_path)
            self.log_message(f"📥 Loaded nodes: {len(self.nodes_df)} | edges: {len(self.edges_df)}")
            self.build_graph()
            self.data_loaded = True
        except Exception as e:
            self.data_loaded = False
            self.log_message(f"❌ Failed to load data: {e}")
            messagebox.showerror("Load Error", f"Failed to load CSV files:\n{e}")
    def build_graph(self):
        self.log_message("🔗 Building graph structure (PyG only)...")

        # Ensure node ID column
        if "ID" not in self.nodes_df.columns:
            if "id" in self.nodes_df.columns:
                self.nodes_df = self.nodes_df.rename(columns={"id": "ID"})
            else:
                self.nodes_df["ID"] = range(len(self.nodes_df))

        all_nodes = self.nodes_df["ID"].astype(int).tolist()
        self.node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}

        # ✅ NEW: Build name→ID mapping if column exists
        print(self.nodes_df.columns)
        if "Name" in self.nodes_df.columns:
            self.name_to_id = dict(zip(self.nodes_df["Name"].astype(str), self.nodes_df["ID"].astype(int)))
        else:
            self.name_to_id = {}
        print(self.name_to_id)
        # Identify toxin candidate nodes (heuristic range)
        self.toxin_nodes = {node for node in all_nodes if 1000001 <= node <= 1003678}

        # Select edge columns
        source_col = "source_id" if "source_id" in self.edges_df.columns else "source"
        target_col = "target_id" if "target_id" in self.edges_df.columns else "target"
        if source_col not in self.edges_df.columns or target_col not in self.edges_df.columns:
            # Try common alternatives before failing
            candidates = ["src", "sourceID", "from", "s"]
            for c in candidates:
                if c in self.edges_df.columns:
                    source_col = c
                    break
            candidates = ["dst", "targetID", "to", "t"]
            for c in candidates:
                if c in self.edges_df.columns:
                    target_col = c
                    break
        if source_col not in self.edges_df.columns or target_col not in self.edges_df.columns:
            raise ValueError("Edges CSV must contain source/target columns (e.g., source,target).")

        # Relation mapping
        if "relation" in self.edges_df.columns:
            unique_relations = pd.Series(self.edges_df["relation"].fillna("default")).unique().tolist()
        else:
            unique_relations = ["default"]
            self.edges_df["relation"] = "default"
        self.relation_to_idx = {rel: idx for idx, rel in enumerate(unique_relations)}

        # Build edges
        edge_list, edge_types = [], []
        for _, row in self.edges_df.iterrows():
            try:
                s, t = int(row[source_col]), int(row[target_col])
                if s in self.node_to_idx and t in self.node_to_idx and s != t:
                    s_idx, t_idx = self.node_to_idx[s], self.node_to_idx[t]
                    edge_list.append([s_idx, t_idx])
                    edge_list.append([t_idx, s_idx])  # undirected mirror
                    rel_idx = self.relation_to_idx.get(row["relation"], 0)
                    edge_types.extend([rel_idx, rel_idx])
            except Exception:
                continue

        # Fallback minimal connectivity
        if not edge_list:
            num_nodes = len(all_nodes)
            edge_list = [[i, i] for i in range(num_nodes)]
            edge_types = [0] * num_nodes

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        x = torch.tensor(all_nodes, dtype=torch.long)

        self.graph_data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        self.log_message(f"✅ Graph built: {len(all_nodes)} nodes, {edge_index.size(1)} edges")
        self.log_message(f"🦠 Identified {len(self.toxin_nodes)} potential toxin nodes")

    # --------------------------
    # Prediction
    # --------------------------
    def _parse_input_nodes(self):
        raw = self.input_nodes_var.get().strip()
        if not raw:
            return []
        ids = []
        for token in raw.split(","):
            token = token.strip()  # ✅ remove extra spaces
            if token == "":
                continue

            # ✅ Case-insensitive match for names
            match = None
            for name, nid in getattr(self, "name_to_id", {}).items():
                if name.lower() == token.lower():
                    match = nid
                    break

            if match is not None:
                ids.append(match)
            else:
                # fallback: try if it's already an ID
                try:
                    tid = int(token)
                    if tid in self.node_to_idx:
                        ids.append(tid)
                except ValueError:
                    self.log_message(f"⚠️ Unrecognized input: {token}")
        return ids


    # def _parse_input_nodes(self):
    #     raw = self.input_nodes_var.get().strip()
    #     if not raw:
    #         return []
    #     ids = []
    #     for token in raw.replace(" ", "").split(","):
    #         if token == "":
    #             continue
    #         try:
    #             ids.append(int(token))
    #         except ValueError:
    #             pass
    #     return ids

    def _subgraph_score(self, model, input_ids, candidate_id):
        """
        Build a tiny subgraph consisting of the input nodes + one candidate,
        using edges filtered from the global graph, then score with the model.
        Returns a float probability in [0,1].
        """
        device = "cpu"
        gd = self.graph_data
        # Keep nodes that exist in the global graph
        keep_ids = [n for n in (input_ids + [candidate_id]) if n in self.node_to_idx]
        if not keep_ids:
            return 0.0

        # Map to local indices 0..k-1
        local_map = {nid: i for i, nid in enumerate(keep_ids)}
        inv_ids = torch.tensor(keep_ids, dtype=torch.long, device=device)

        # Filter edges where both endpoints are in keep_ids
        global_ei = gd.edge_index
        src = global_ei[0].tolist()
        dst = global_ei[1].tolist()

        # Convert global node indices back to node IDs via idx_to_node
        # idx_to_node maps local index -> nodeID in the global graph
        # We need a quick inverse: global index -> nodeID
        global_idx_to_node = self.idx_to_node

        filt_edges = []
        filt_types = []
        for e_idx in range(len(src)):
            g_u = src[e_idx]
            g_v = dst[e_idx]
            nid_u = global_idx_to_node.get(g_u)
            nid_v = global_idx_to_node.get(g_v)
            if nid_u in local_map and nid_v in local_map:
                u = local_map[nid_u]
                v = local_map[nid_v]
                filt_edges.append([u, v])
                if hasattr(gd, "edge_type") and gd.edge_type is not None and gd.edge_type.numel() > e_idx:
                    filt_types.append(int(gd.edge_type[e_idx].item()))
                else:
                    filt_types.append(0)

        # If no edges, create self-loops to keep layers stable
        if not filt_edges:
            for i in range(len(keep_ids)):
                filt_edges.append([i, i])
                filt_types.append(0)

        edge_index = torch.tensor(filt_edges, dtype=torch.long, device=device).t().contiguous()
        edge_type = torch.tensor(filt_types, dtype=torch.long, device=device)

        x = inv_ids.to(device)

        # batch: single graph
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            if isinstance(model, RGCNModel):
                out = model(x, edge_index, edge_type=edge_type, batch=batch)
            else:
                out = model(x, edge_index, batch=batch)
            prob = float(out.view(-1).mean().item())
        return max(0.0, min(1.0, prob))

    def run_prediction(self):
        try:
            if not self.data_loaded or self.graph_data is None:
                messagebox.showwarning("No Data", "Please load nodes/edges CSVs first.")
                return
            if not self.models:
                messagebox.showwarning("No Models", "Please load or select a trained model.")
                return
            model_name = self.selected_model_var.get()
            if not model_name or model_name not in self.models:
                messagebox.showwarning("No Model Selected", "Please select a model from the dropdown.")
                return

            input_ids = self._parse_input_nodes()
            if not input_ids:
                messagebox.showwarning("Input Required", "Please enter one or more node IDs.")
                return

            # Filter input IDs to those present
            input_ids = [n for n in input_ids if n in self.node_to_idx]
            if not input_ids:
                messagebox.showwarning("Not Found", "None of the provided node IDs exist in the graph.")
                return

            model, meta = self.models[model_name]
            topk = int(self.topk_var.get())
            self.log_message(f"🚀 Running prediction with model: {model_name} | TopK={topk}")

            # Candidates: all toxin nodes that are NOT in the inputs
            candidates = [n for n in sorted(self.toxin_nodes) if n not in input_ids]
            if not candidates:
                self.log_message("⚠️ No toxin candidates found by the current heuristic range.")
                messagebox.showinfo("No Candidates", "No toxin candidates found (range 1000001–1003678).")
                return

            # Score each candidate by subgraph with inputs + candidate
            scores = []
            for cid in candidates:
                p = self._subgraph_score(model, input_ids, cid)
                scores.append((cid, p))

            # Sort and take Top-K
            scores.sort(key=lambda x: x[1], reverse=True)
            top = scores[:topk]

            # Display
            for i in self.results_tree.get_children():
                self.results_tree.delete(i)
            self.last_results = []

            for rank, (cid, prob) in enumerate(top, start=1):
                node_type = "Toxin"
                toxin_name = ""
                if self.nodes_df is not None and "ID" in self.nodes_df.columns and "Name" in self.nodes_df.columns:
                    match = self.nodes_df.loc[self.nodes_df["ID"] == cid, "Name"]
                    if not match.empty:
                        toxin_name = match.values[0]

                self.results_tree.insert("", tk.END, values=(rank, cid, toxin_name, f"{prob*100:.2f}", node_type))
                self.last_results.append({
                    "Rank": rank,
                    "ToxinID": cid,
                    "ToxinName": toxin_name,
                    "Confidence": prob,
                    "Type": node_type,
                })

            self.log_message(f"✅ Prediction done. Displaying Top-{len(top)}.")
            self.notebook.select(self.results_tab)

        except Exception as e:
            self.log_message(f"❌ Prediction failed: {e}")
            traceback.print_exc()
            messagebox.showerror("Prediction Error", f"An error occurred:\n{e}")

    # --------------------------
    # Export
    # --------------------------
    def export_results(self):
        if not self.last_results:
            messagebox.showinfo("No Results", "There are no results to export yet.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="toxin_predictions.csv",
        )
        if not out_path:
            return
        try:
            df = pd.DataFrame(self.last_results)
            df.to_csv(out_path, index=False)
            self.log_message(f"💾 Results saved to: {out_path}")
            messagebox.showinfo("Exported", f"Results saved to:\n{out_path}")
        except Exception as e:
            self.log_message(f"❌ Export failed: {e}")
            messagebox.showerror("Export Error", f"Failed to save CSV:\n{e}")


# ==========================
# Main
# ==========================
if __name__ == "__main__":
    root = tk.Tk()
    app = ToxinPredictionApp(root)

    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication closed by user.")
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Critical Error", f"Application encountered an error: {e}")
