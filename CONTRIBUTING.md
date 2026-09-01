# Contributing to AttnLab

AttnLab welcomes focused contributions that make attention mechanisms easier
to inspect, compare, and verify.

## Before opening a pull request

1. Open an issue for architectural changes or new attention variants.
2. Keep the IR, runtime, trace, API, and visualization layers decoupled.
3. Label educational approximations explicitly. Do not claim paper fidelity
   without a reference equation and numerical parity test.
4. Keep changes scoped. Avoid unrelated refactors in feature pull requests.

## Development setup

```bash
make setup
```

Run the backend and frontend in separate terminals:

```bash
make backend
make frontend
```

The application is available at <http://127.0.0.1:5173>.

## Quality checks

Run all checks before submitting:

```bash
make lint
make test
```

Use `make format` to apply the repository formatters.

## Adding an attention mechanism

A new mechanism should include:

- NumPy runtime operations with unit tests.
- IR nodes and trace events for every meaningful calculation.
- Prefill and decode behavior where applicable.
- Persistent state represented through the generic `MemorySpec`.
- Node Inspector descriptions and a safe tensor visualization.
- A paper or specification reference in `docs/ALGORITHMS.md`.
- An explicit fidelity level and a list of omitted production details.

## Pull request expectations

- Explain the user-visible behavior and mathematical change.
- Include tests for shapes, numerical behavior, and persistent state.
- Add screenshots for visual changes.
- Keep generated files and local environments out of the commit.

By contributing, you agree that your contribution is licensed under the
Apache License 2.0.
