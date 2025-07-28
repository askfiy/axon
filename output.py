from pathlib import Path

# 配置忽略的目录和文件
IGNORE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    "tests",
    ".env",
    "versions",
    "alembic",
}
IGNORE_FILES = {
    "uv.lock",
    "pyproject.toml",
    "output.py",
    ".gitignore",
    ".python-version",
    "alembic.ini",
}

# 输出文件名
OUTPUT_FILE = "all_code.md"

# 支持的语言后缀到 markdown 语言名的映射
EXTENSION_LANG_MAP = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".json": "json",
    ".html": "html",
    ".css": "css",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
    ".md": "markdown",
    ".sql": "sql",
    ".lua": "lua",
}


def get_language_for_file(file_path: Path) -> str:
    return EXTENSION_LANG_MAP.get(file_path.suffix, "")


def should_ignore(path: Path) -> bool:
    return any(part in IGNORE_DIRS for part in path.parts) or path.name in IGNORE_FILES


def collect_code_files(root: Path):
    return [f for f in root.rglob("*") if f.is_file() and not should_ignore(f)]


def write_markdown(files: list[Path], output_path: Path):
    with output_path.open("w", encoding="utf-8") as out_file:
        for file in sorted(files):
            lang = get_language_for_file(file)
            out_file.write(f"\n## `{file}`\n\n")
            out_file.write(f"```{lang}\n")
            try:
                content = file.read_text(encoding="utf-8")
            except Exception as e:
                content = f"# Error reading file: {e}"
            out_file.write(content)
            out_file.write("\n```\n")


def main():
    project_root = Path(".").resolve()
    files = collect_code_files(project_root)
    write_markdown(files, project_root / OUTPUT_FILE)
    print(f"✅ Code successfully written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
