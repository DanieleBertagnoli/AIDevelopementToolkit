from pathlib import Path

import mkdocs_gen_files

PACKAGE = "aidevelopementtoolkit"

nav = mkdocs_gen_files.Nav()

for path in sorted(Path(PACKAGE).rglob("*.py")):

    module_path = path.relative_to(PACKAGE).with_suffix("")

    parts = module_path.parts

    # Skip executable modules
    if parts[-1] == "__main__":
        continue

    # Convert __init__.py into the package itself
    if parts[-1] == "__init__":
        module_path = module_path.parent
        parts = module_path.parts
        full_doc_path = Path(PACKAGE, *module_path.parts, "index.md")
    else:
        full_doc_path = Path(PACKAGE, *module_path.parts).with_suffix(".md")

    # Add to SUMMARY.md navigation
    nav[parts] = full_doc_path.relative_to(PACKAGE).as_posix()

    # Generate API page
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join((PACKAGE,) + parts)
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open(f"{PACKAGE}/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())