import os

import nbformat


def fix_widget_metadata(nb):
  changed = False
  metadata = nb.get('metadata', {})
  if 'colab' not in metadata:
    return changed
  widgets = metadata.get('widgets', {})
  widgets['state'] = {}
  changed = True
  widget_state = widgets.get('application/vnd.jupyter.widget-state+json', {})
  for widget_id, widget_obj in widget_state.items():
    if isinstance(widget_obj, dict) and 'state' not in widget_obj:
      widget_obj['state'] = {}
      changed = True
  return changed


def main():
  print("Start")
  fixed_files = []
  for root, _, files in os.walk('.'):
    for file in files:
      if file.endswith('.ipynb'):
        path = os.path.join(root, file)
        with open(path, encoding='utf-8') as notebook_file:
          nb = nbformat.read(notebook_file, as_version=4)
        if fix_widget_metadata(nb):
          with open(path, 'w', encoding='utf-8') as notebook_file:
            nbformat.write(nb, notebook_file)
          fixed_files.append(file)
  print(f"Find {len(fixed_files)} files")
  for file in fixed_files:
    print(f"{file}")


if __name__ == '__main__':
  main()
