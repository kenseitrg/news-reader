# Agent Working Practices

## Tooling

- **Package & venv management**: `uv` (not pip, not pipenv, not poetry)
- **Linting**: `ruff`
- **Type checking**: `pyright` (via `uv tool run pyright` or `npx pyright`)
- **Formatting**: `ruff format`
- **Testing**: `pytest`

## Code Conventions

### Style & Quality

- Follow industry best practices and clean code conventions.
- Keep functions small and focused — single responsibility.
- Use descriptive names for variables, functions, and classes.
- Avoid unnecessary comments in code — let the code speak.
- Use strict type annotations everywhere.

### Docstrings

- Use **Google-style** docstrings for all public modules, classes, and functions.
- Example:
  ```python
  def fetch_articles(source_id: int, limit: int = 10) -> list[dict]:
      """Fetch articles from a given source.

      Args:
          source_id: The database ID of the source.
          limit: Maximum number of articles to return.

      Returns:
          A list of article dictionaries.
      """
  ```

### Type Annotations

- Always annotate function signatures (parameters and return types).
- Prefer `list[X]` over `List[X]`, `dict[K, V]` over `Dict[K, V]` (built-in generics).
- Use `|` for union types (Python 3.11+), e.g. `str | None`.

## Pre-Commit Checklist

Before submitting any changes, run:

```bash
uv run ruff check src/
uv run ruff format --check src/
uv run pyright src/
```

If the project has tests:

```bash
uv run pytest
```

## Commands Quick Reference

```bash
uv add <package>          # add dependency
uv remove <package>       # remove dependency
uv run <script>           # run script in venv
uv run ruff check src/    # lint
uv run ruff format src/   # format
uv run pyright src/       # type check
uv run pytest             # run tests
```
