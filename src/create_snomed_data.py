import os
import re
import shutil
import sys

def extract_snomed_version(folder_name : str):
    """Extracts the SNOMED version from the folder name using a regular expression.
    
    Parameters:
        folder_name (str):
            String containing the name of the folder, which should include the SNOMED version in the format "_YYYYMMDDT".
    
    """
    match = re.search(r'_(\d{8})T', folder_name)
    if not match:
        raise ValueError("Could not extract SNOMED version from folder name.")
    return match.group(1)

if len(sys.argv) < 3:
    raise ValueError("Invalid number of arguments." \
    "\n Usage: python extract_snomed.py RF2_INT_path" \
    "\n Usage: python extract_snomed.py RF2_INT_path RF2_ES_path")

rf2_int_path = os.path.abspath(sys.argv[1])
rf2_es_path = os.path.abspath(sys.argv[2]) if len(sys.argv) == 3 else None

# Obtain the folder name to extract the version
folder_name = os.path.basename(rf2_int_path.rstrip(os.sep))

# Extract version
version = extract_snomed_version(folder_name)
print(f"Detected SNOMED version: {version}")

# Source directory
int_terminology_path = os.path.join(rf2_int_path, "Snapshot", "Terminology")

if not os.path.exists(int_terminology_path):
    raise FileNotFoundError(f"Terminology folder not found: {int_terminology_path}")

# Output directory
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_dir = os.path.join(repo_root, "snomed_data")
os.makedirs(output_dir, exist_ok=True)

# File mappings
file_map = {
    f"sct2_Concept_Snapshot_INT_{version}.txt": f"conceptInternational_{version}.txt",
    f"sct2_Relationship_Snapshot_INT_{version}.txt": f"relationshipInternational_{version}.txt",
    f"sct2_Description_Snapshot_INT_{version}.txt": f"descriptionInternational_{version}.txt",
}

# Copy and rename files
for src_name, dst_name in file_map.items():
    src_path = os.path.join(int_terminology_path, src_name)
    dst_path = os.path.join(output_dir, dst_name)

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Missing expected file: {src_path}")

    shutil.copy2(src_path, dst_path)
    print(f"Copied: {src_name} -> {dst_name}")

if rf2_es_path:
    es_folder_name = os.path.basename(rf2_es_path.rstrip(os.sep))
    es_version = extract_snomed_version(es_folder_name)

    if es_version != version:
        raise ValueError(
            f"Version mismatch: INT={version}, ES={es_version}"
        )

    es_terminology = os.path.join(rf2_es_path, "Snapshot", "Terminology")

    if not os.path.exists(es_terminology):
        raise FileNotFoundError(f"Spanish Terminology folder not found: {es_terminology}")

    es_src_name = f"sct2_Description_SpanishExtensionSnapshot-es_INT_{version}.txt"
    es_dst_name = f"conceptsSpanish_{version}.txt"

    es_src = os.path.join(es_terminology, es_src_name)
    es_dst = os.path.join(output_dir, es_dst_name)

    if not os.path.exists(es_src):
        raise FileNotFoundError(f"Missing Spanish file: {es_src}")

    shutil.copy2(es_src, es_dst)
    print(f"Copied ES: {es_src_name} -> {es_dst_name}")


print(f"\nAll files copied successfully to: {output_dir}")

