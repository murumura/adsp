"""Keep Jupyter's displayed results out of saved notebook source files."""


def clear_outputs_before_save(model, **kwargs):
  """Jupyter ContentsManager pre-save hook; displayed kernel outputs remain in the UI."""
  if model.get("type") != "notebook":
    return
  notebook = model.get("content", {})
  if notebook.get("nbformat") != 4:
    return
  for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
      cell["outputs"] = []
      cell["execution_count"] = None
      cell.get("metadata", {}).pop("execution", None)
  notebook.get("metadata", {}).pop("widgets", None)
