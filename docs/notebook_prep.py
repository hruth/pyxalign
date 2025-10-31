import argparse
import json
import base64
from pathlib import Path

def convert_notebook_attachments(notebook_path: str, output_path: str):
    """Convert notebook attachments to embedded base64 images."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'markdown' and 'attachments' in cell:
            attachments = cell['attachments']
            source = cell['source']
            
            # Convert source to list if it's a string
            if isinstance(source, str):
                source = [source]
            
            new_source = []
            for line in source:
                # Replace attachment references with data URIs
                for att_name, att_data in attachments.items():
                    if f'attachment:{att_name}' in line:
                        # Get the first available image format
                        mime_type = list(att_data.keys())[0]
                        data = att_data[mime_type]
                        data_uri = f'data:{mime_type};base64,{data}'
                        line = line.replace(f'attachment:{att_name}', data_uri)
                new_source.append(line)
            
            cell['source'] = new_source
            # Remove attachments after conversion
            del cell['attachments']
    
    # Save the modified notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Converted notebook saved to: {output_path}")

# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebook_path",
    )
    parser.add_argument(
        "output_path",
    )
    args = parser.parse_args()

    convert_notebook_attachments(args.notebook_path, args.output_path)
    
if __name__ == "__main__":
    main()
