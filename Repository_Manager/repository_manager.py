import os
import time
import json
import hashlib
import ast
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional


class RepoManager:
    """
    ローカルリポジトリを管理・監視し、状態をMarkdownレポートとして出力するクラス
    AST解析(Python) + 構造解析(Markdown) 機能付き
    """

    def __init__(self, root_path: str = "."):
        # パスの解決と存在確認
        self.root_path = Path(root_path).resolve()
        if not self.root_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_path}")

        self.state_file = self.root_path / ".repo_manager_state.json"
        self.report_file = self.root_path / "PROJECT_STATUS.md"

        # 無視するディレクトリ・ファイル
        self.ignore_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            "vendor",
            ".idea",
            ".vscode",
            "build",
            "dist",
            "env",
            "venv",
            ".venv311",
            "google-cloud-sdk",
        }
        self.ignore_files = {
            ".DS_Store",
            "Thumbs.db",
            self.state_file.name,
            self.report_file.name,
            os.path.basename(__file__),
        }

        self.file_types = {
            ".py": "Python Script",
            ".js": "JavaScript File",
            ".ts": "TypeScript File",
            ".php": "PHP Script",
            ".html": "HTML Template",
            ".css": "Stylesheet",
            ".md": "Documentation",
            ".json": "Config/Data",
        }

    def _calculate_hash(self, file_path: Path) -> str:
        """ファイルのハッシュ値を計算"""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def _analyze_python_content(self, file_path: Path) -> Dict:
        """Pythonファイルの中身をASTで解析"""
        definitions = {"classes": [], "functions": []}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    doc = ast.get_docstring(node) or "No description"
                    doc_summary = doc.split("\n")[0]
                    definitions["classes"].append(
                        {"name": node.name, "methods": methods, "doc": doc_summary}
                    )
                elif isinstance(node, ast.FunctionDef):
                    if not hasattr(node, "is_method"):
                        pass
        except Exception as e:
            print(f"Warning: Could not parse Python {file_path}: {e}")

        return definitions

    def _analyze_markdown_content(self, file_path: Path) -> List[Dict]:
        """
        Markdownファイルのヘッダー(#)を抽出して構造化する
        """
        headers = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        # #の数と内容を分離
                        parts = stripped.split(" ", 1)
                        if parts:
                            # 見出しレベル (# の数)
                            level = len(parts[0])
                            # 全てが # であることを確認（コメントアウトなどを誤検知しないため）
                            if all(char == "#" for char in parts[0]):
                                text = parts[1] if len(parts) > 1 else ""
                                headers.append({"level": level, "text": text})
        except Exception as e:
            print(f"Warning: Could not parse Markdown {file_path}: {e}")
        return headers

    def _get_file_info(self, file_path: Path) -> Dict:
        """ファイルのメタデータと構造解析を取得"""
        stat = file_path.stat()
        info = {
            "path": str(file_path.relative_to(self.root_path)),
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "hash": self._calculate_hash(file_path),
            "type": self.file_types.get(file_path.suffix, "Unknown"),
            "structure": {},
        }

        # 拡張子に応じて解析処理を分岐
        if file_path.suffix == ".py":
            info["structure"] = self._analyze_python_content(file_path)
        elif file_path.suffix == ".md" and file_path.name != self.report_file.name:
            # レポートファイル自身は解析しない（無限ループ防止＆ノイズ除去）
            info["structure"] = {"headers": self._analyze_markdown_content(file_path)}

        return info

    def scan_repository(self) -> Dict[str, Dict]:
        current_state = {}
        print(f"Scanning directory: {self.root_path} ...")

        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            for file in files:
                if file in self.ignore_files:
                    continue
                file_path = Path(root) / file
                current_state[str(file_path.relative_to(self.root_path))] = self._get_file_info(
                    file_path
                )
        return current_state

    def load_previous_state(self) -> Dict:
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_state(self, state: Dict):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def compare_states(self, old_state: Dict, new_state: Dict) -> Dict:
        added = sorted([k for k in new_state.keys() if k not in old_state])
        deleted = sorted([k for k in old_state.keys() if k not in new_state])
        modified = sorted(
            [
                k
                for k in new_state.keys()
                if k in old_state and new_state[k]["hash"] != old_state[k]["hash"]
            ]
        )
        return {"added": added, "modified": modified, "deleted": deleted}

    def generate_tree_structure(self, path: Path, prefix: str = "") -> str:
        output = ""
        try:
            entries = sorted(
                [
                    e
                    for e in path.iterdir()
                    if e.name not in self.ignore_dirs and e.name not in self.ignore_files
                ]
            )
        except PermissionError:
            return ""

        pointers = [("├── ", "│   ")] * (len(entries) - 1) + [("└── ", "    ")] if entries else []
        for pointer, entry in zip(pointers, entries):
            output += f"{prefix}{pointer[0]}{entry.name}"
            if entry.is_dir():
                output += "/\n" + self.generate_tree_structure(entry, prefix + pointer[1])
            else:
                output += "\n"
        return output

    def generate_mermaid_diagram(self, current_state: Dict) -> str:
        mermaid = "```mermaid\nclassDiagram\n"
        has_content = False
        for path, info in current_state.items():
            structure = info.get("structure", {})
            classes = structure.get("classes", [])
            if classes:
                has_content = True
                file_name = Path(path).name
                for cls in classes:
                    class_name = cls["name"]
                    doc = cls["doc"].replace('"', "'")
                    mermaid += f"    class {class_name} {{\n        %% {doc}\n"
                    for method in cls["methods"]:
                        mermaid += f"        +{method}()\n"
                    mermaid += "    }\n"
                    mermaid += f'    note for {class_name} "File: {file_name}"\n'
        mermaid += "```\n"
        if not has_content:
            return "*No Python classes detected for visualization.*\n"
        return mermaid

    def generate_docs_overview(self, current_state: Dict) -> str:
        """Markdownドキュメントの構造一覧を生成"""
        output = "## 4. Documentation Index (ドキュメント構成図)\n"
        output += "リポジトリ内のMarkdownファイルの見出し一覧です。\n\n"

        has_docs = False
        for path, info in sorted(current_state.items()):
            # Documentationタイプで、headers情報がある場合
            if info["type"] == "Documentation" and "headers" in info.get("structure", {}):
                headers = info["structure"]["headers"]
                if not headers:
                    continue

                has_docs = True
                output += f"### 📄 `{path}`\n"
                # 折りたたみ式の詳細ブロックを作成
                output += "<details><summary>Click to show content structure</summary>\n\n"
                for h in headers:
                    # 見出しレベルに応じてインデント (Markdownのリスト形式)
                    indent = "  " * (max(0, h["level"] - 1))
                    output += f"{indent}- {h['text']}\n"
                output += "\n</details>\n\n"

        if not has_docs:
            return ""
        return output

    def generate_markdown_report(self, changes: Dict, current_state: Dict):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        content = f"# 📊 Project Status Report\n\n"
        content += f"**Path:** `{self.root_path}`\n"
        content += f"**Updated:** {timestamp}\n\n"

        # 1. Visual Overview
        content += "## 1. Architecture Overview (Class Diagram)\n"
        content += self.generate_mermaid_diagram(current_state)
        content += "\n---\n\n"

        # 2. Recent Changes
        content += "## 2. Recent Changes\n"
        if not any(changes.values()):
            content += "No changes detected.\n"
        else:
            if changes["added"]:
                content += "### ✨ Added\n" + "".join([f"- `{f}`\n" for f in changes["added"]])
            if changes["modified"]:
                content += "### 📝 Modified\n" + "".join(
                    [f"- `{f}`\n" for f in changes["modified"]]
                )
            if changes["deleted"]:
                content += "### 🗑️ Deleted\n" + "".join([f"- `{f}`\n" for f in changes["deleted"]])
        content += "\n---\n\n"

        # 3. Structure
        content += "## 3. Project Tree\n"
        content += "```text\n" + self.generate_tree_structure(self.root_path) + "```\n\n"
        content += "---\n\n"

        # 4. Docs Overview (New!)
        content += self.generate_docs_overview(current_state)

        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Report generated: {self.report_file}")

    def run(self):
        print(f"--- Analyzing Repository: {self.root_path} ---")
        try:
            new_state = self.scan_repository()
            old_state = self.load_previous_state()
            changes = self.compare_states(old_state, new_state)
            self.generate_markdown_report(changes, new_state)
            self.save_state(new_state)
            print("--- Done ---")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="Repository Status Manager")
    parser.add_argument(
        "path", nargs="?", default=".", help="Target repository path (default: current directory)"
    )
    args = parser.parse_args()

    # 指定されたパス（またはデフォルト）で実行
    RepoManager(args.path).run()
