"""Generate the code reference pages and navigation."""

import importlib.util
from pathlib import Path

import mkdocs_gen_files


def can_import_module(module_name: str) -> bool:
    """Check if a module can be imported without errors."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


nav = mkdocs_gen_files.Nav()

src = Path(__file__).parent.parent
package_dir = src / "ccproxy"

# Modules to skip due to known issues
SKIP_MODULES = {
    "ccproxy.api.dependencies",  # Has parameter annotation issues
}

# Skip entire directories that have issues
SKIP_PATTERNS = {
    "ccproxy.services.http",  # HTTP service modules have import/annotation issues
}

for path in sorted(package_dir.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    # Skip private modules
    if any(part.startswith("_") and part != "__init__" for part in parts):
        continue

    # Check if module is in skip list
    module_name = ".".join(parts)
    if module_name in SKIP_MODULES:
        continue

    # Check if module matches skip patterns
    skip_module = False
    for pattern in SKIP_PATTERNS:
        if module_name.startswith(pattern):
            skip_module = True
            break

    if skip_module:
        continue

    # Check if module can be imported
    if not can_import_module(module_name):
        print(f"Skipping module that cannot be imported: {module_name}")
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# {ident}\n\n")
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(src))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
