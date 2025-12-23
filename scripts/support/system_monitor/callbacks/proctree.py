if True:
    from typing import Any
    from datetime import datetime
    import dash
    from dash import dcc
    from dash.dependencies import Output, Input, State
    import psutil

    def register_proctree_callbacks(app: Any, *, cy_available: bool) -> None:
        # PID search: suggest processes matching a search string (similar to `ps -AFl | grep <pattern>`)
        @app.callback(
            Output("proctree-suggestions", "options"),
            [Input("proctree-search", "value")],
            prevent_initial_call=False,
        )
        def update_proctree_suggestions(search_text):
            opts = []
            try:
                pattern = (search_text or "").strip()
                if not pattern:
                    return []
                pattern_low = pattern.lower()
                # Collect processes with safe attribute access
                matches = []
                for p in psutil.process_iter(attrs=["pid", "name", "username", "cmdline", "num_threads"]):
                    try:
                        info = p.info
                        pid = info.get("pid")
                        name = info.get("name") or ""
                        username = info.get("username") or ""
                        cmdline_list = info.get("cmdline") or []
                        cmdline = " ".join(cmdline_list)
                        haystack = f"{name} {cmdline}".lower()
                        if pattern_low in haystack:
                            threads = info.get("num_threads") or 0
                            label = f"PID {pid} • {username} • thr={threads} • {name} — {cmdline}".strip()
                            matches.append({"label": label[:300], "value": pid})
                    except Exception:
                        continue
                # Limit to first 50
                opts = matches[:50]
            except Exception:
                opts = []
            return opts

        # When a suggestion is selected, set the PID input
        @app.callback(
            Output("proctree-pid", "value"),
            [Input("proctree-suggestions", "value")],
            prevent_initial_call=True,
        )
        def set_pid_from_selection(selected_pid):
            try:
                if selected_pid is None:
                    return dash.no_update
                return int(selected_pid)
            except Exception:
                return dash.no_update

        # Theme-aware high-contrast styles for PID search controls
        @app.callback(
            Output("proctree-suggestions", "style"),
            [Input("theme-toggle", "value")],
        )
        def style_proctree_dropdown(theme_value):
            light = {
                "width": "420px",
                "display": "inline-block",
                "color": "#111",
                "backgroundColor": "#ffffff",
                "border": "1px solid #888",
            }
            dark = {
                "width": "420px",
                "display": "inline-block",
                "color": "#eee",
                "backgroundColor": "#222",
                "border": "1px solid #555",
            }
            return dark if theme_value == "dark" else light

        @app.callback(
            Output("proctree-search", "style"),
            [Input("theme-toggle", "value")],
        )
        def style_proctree_search(theme_value):
            base = {"width": "320px", "marginLeft": "6px", "marginRight": "8px"}
            if theme_value == "dark":
                base.update({"backgroundColor": "#222", "color": "#eee", "border": "1px solid #555"})
            else:
                base.update({"backgroundColor": "#fff", "color": "#111", "border": "1px solid #888"})
            return base

        # Toggle between text and graph tree containers
        @app.callback(
            [Output("proctree-cyto-container", "style"), Output("proctree-tree-text-container", "style")],
            [Input("proctree-view-mode", "value")],
        )
        def toggle_tree_view(view_mode):
            # Always show the Graph container when Graph is selected, even if dash-cytoscape
            # is not installed, so the fallback help message is visible.
            if view_mode == "graph":
                return {"display": "block"}, {"display": "none"}
            # default to text view
            return {"display": "none"}, {"display": "block"}

        # Build cytoscape elements from last summary
        if cy_available:

            @app.callback(
                Output("proctree-graph", "elements"),
                [Input("proctree-last-summary", "data")],
            )
            def build_cytoscape_elements(summary):
                try:
                    plist = (summary or {}).get("processes", [])
                    if not plist:
                        return []
                    nodes = []
                    edges = []
                    pids = set()
                    for p in plist:
                        pid = p.get("pid")
                        name = p.get("name") or "?"
                        threads = int(p.get("threads") or 0)
                        pids.add(pid)
                        nodes.append(
                            {
                                "data": {
                                    "id": str(pid),
                                    "label": f"{name}({pid}) t={threads}",
                                    "threads": threads,
                                }
                            }
                        )
                    for p in plist:
                        pid = p.get("pid")
                        ppid = p.get("ppid")
                        if ppid in pids and pid in pids and ppid is not None and pid is not None:
                            edges.append({"data": {"source": str(ppid), "target": str(pid)}})
                    return nodes + edges
                except Exception:
                    return []

            @app.callback(
                Output("proctree-graph", "stylesheet"),
                [Input("theme-toggle", "value")],
            )
            def cytoscape_stylesheet(theme_value):
                # map threads to size/color
                node_color_dark = "#4aa3ff"
                node_color_light = "#1f77b4"
                text_color_dark = "#e5e5e5"
                text_color_light = "#222222"
                edge_color_dark = "#888"
                edge_color_light = "#aaa"
                base = [
                    {
                        "selector": "node",
                        "style": {
                            "label": "data(label)",
                            "font-size": 10,
                            "color": (text_color_dark if theme_value == "dark" else text_color_light),
                            "background-color": (node_color_dark if theme_value == "dark" else node_color_light),
                            "width": "mapData(threads, 0, 64, 20, 60)",
                            "height": "mapData(threads, 0, 64, 20, 60)",
                            "text-valign": "center",
                            "text-halign": "center",
                        },
                    },
                    {
                        "selector": "edge",
                        "style": {
                            "line-color": (edge_color_dark if theme_value == "dark" else edge_color_light),
                            "target-arrow-color": (edge_color_dark if theme_value == "dark" else edge_color_light),
                            "target-arrow-shape": "triangle",
                            "curve-style": "bezier",
                            "width": 1.5,
                        },
                    },
                ]
                return base

            @app.callback(
                Output("proctree-node-details", "children"),
                [Input("proctree-graph", "tapNodeData")],
                [State("proctree-last-summary", "data")],
            )
            def show_node_details(tap_node, summary):
                try:
                    if not tap_node:
                        return ""
                    pid = int(tap_node.get("id"))
                    plist = (summary or {}).get("processes", [])
                    ent = next((p for p in plist if p.get("pid") == pid), None)
                    if not ent:
                        return ""
                    name = ent.get("name") or "?"
                    ppid = ent.get("ppid")
                    threads = ent.get("threads")
                    return f"Selected: {name} ({pid}) — PPID={ppid}, Threads={threads}"
                except Exception:
                    return ""

            # Force a layout re-run whenever elements change
            @app.callback(
                Output("proctree-graph", "layout"),
                [Input("proctree-graph", "elements")],
            )
            def refresh_cyto_layout(elements):
                return {"name": "breadthfirst", "directed": True}

        # Status helper under the graph container
        @app.callback(
            Output("proctree-graph-status", "children"),
            [Input("proctree-view-mode", "value"), Input("proctree-last-summary", "data")],
        )
        def update_graph_status(view_mode, summary):
            if view_mode != "graph":
                return ""
            if not cy_available:
                return "Graph view requires dash-cytoscape. Install with: pip install dash-cytoscape"
            plist = (summary or {}).get("processes", [])
            if not plist:
                return "No graph data yet. Click 'Inspect' after entering a valid PID."
            # count edges
            by_ppid = {}
            for p in plist:
                by_ppid.setdefault(p.get("ppid"), []).append(p)
            edge_count = sum(len(v) for k, v in by_ppid.items() if k is not None)
            return f"Graph ready: {len(plist)} node(s), {edge_count} edge(s). Tip: click a node to see details."

        # Snapshot current summary
        @app.callback(
            [Output("proctree-snapshot", "data"), Output("proctree-snapshot-status", "children")],
            [Input("proctree-snapshot-btn", "n_clicks")],
            [State("proctree-last-summary", "data")],
            prevent_initial_call=True,
        )
        def take_snapshot(n_clicks, summary):
            try:
                if not summary or not summary.get("processes"):
                    return dash.no_update, "No current tree to snapshot. Run Inspect first."
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                snap = {"timestamp": ts, "summary": summary}
                return snap, f"Snapshot captured at {ts}."
            except Exception:
                return dash.no_update, "Failed to capture snapshot."

        # Diff current summary to snapshot
        @app.callback(
            Output("proctree-diff-result", "children"),
            [Input("proctree-diff-btn", "n_clicks")],
            [State("proctree-snapshot", "data"), State("proctree-last-summary", "data")],
            prevent_initial_call=True,
        )
        def diff_to_snapshot(n_clicks, snapshot, current):
            try:
                if not snapshot or not snapshot.get("summary"):
                    return "No snapshot available. Click 'Take Snapshot' after Inspect."
                snap = snapshot.get("summary") or {}
                snap_plist = snap.get("processes", [])
                cur_plist = (current or {}).get("processes", [])
                snap_by_pid = {p.get("pid"): p for p in snap_plist}
                cur_by_pid = {p.get("pid"): p for p in cur_plist}
                added = sorted([pid for pid in cur_by_pid.keys() if pid not in snap_by_pid])
                removed = sorted([pid for pid in snap_by_pid.keys() if pid not in cur_by_pid])
                changed = []
                for pid in set(cur_by_pid.keys()).intersection(snap_by_pid.keys()):
                    t0 = int(snap_by_pid[pid].get("threads") or 0)
                    t1 = int(cur_by_pid[pid].get("threads") or 0)
                    if t0 != t1:
                        changed.append((pid, t0, t1))
                lines = []
                lines.append(f"Added: {len(added)}")
                if added:
                    lines.extend([f"  + {pid}" for pid in added[:50]])
                    if len(added) > 50:
                        lines.append("  …")
                lines.append(f"Removed: {len(removed)}")
                if removed:
                    lines.extend([f"  - {pid}" for pid in removed[:50]])
                    if len(removed) > 50:
                        lines.append("  …")
                lines.append(f"Thread changes: {len(changed)}")
                if changed:
                    for pid, t0, t1 in changed[:50]:
                        lines.append(f"  ~ {pid}: {t0} -> {t1}")
                    if len(changed) > 50:
                        lines.append("  …")
                return dcc.Markdown("```text\n" + "\n".join(lines) + "\n```")
            except Exception:
                return "Diff failed."
