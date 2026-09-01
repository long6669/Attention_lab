# Changelog

All notable changes to AttnLab are documented in this file. The project follows
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- Side-by-side comparison for two to four attention architectures.
- Synchronized decode and memory-growth projections in Compare Mode.
- Shareable experiment URLs and Graph, Trace, and PNG exports.
- Guided architecture lessons and an examples gallery.
- Shape-based execution metrics for FLOPs, graph size, and cache growth.
- Playwright coverage for comparison, CSA trace playback, and mobile layout.
- Cross-architecture parity between incremental decode and full prefill.

### Changed

- Session storage now uses TTL expiration, LRU eviction, and per-session locks.
- ELK layout and PNG export dependencies load only when needed.
