# Changelog

All notable changes to the tolgee plugin will be documented in this file.

## [1.0.2] - 2026-04-26

### Security
- Added `overrides` in MCP server `package.json` to force-resolve transitive
  dependencies flagged by `npm audit`. Result: **0 vulnerabilities** (down from 7).
  - `@hono/node-server` → 1.19.14 (auth bypass via encoded slashes, middleware bypass)
  - `hono` → 4.12.15 (multiple CVEs: cookie injection, SSE CR/LF, prototype pollution, IPv6 matching)
  - `ajv` → 8.20.0 (ReDoS via `$data`)
  - `express-rate-limit` → 8.4.1 (IPv4-mapped IPv6 bypass)
  - `follow-redirects` → 1.16.0 (auth header leak on cross-domain redirect)
  - `path-to-regexp` → 8.4.2 (multiple ReDoS)
  - `qs` → 6.15.1 (arrayLimit DoS via comma parsing)

## [1.0.1] - 2026-04-26

### Changed
- Bumped and pinned MCP server dependencies for security:
  - `@modelcontextprotocol/sdk` 1.12.0 → 1.29.0
  - `axios` 1.7.0 → 1.15.2 (fixes CVE-2026-40175 SSRF, CVE-2025-62718 NO_PROXY bypass)
  - `zod` 3.23.0 → 3.25.76
  - `@types/node` 22.10.0 → 24.12.2
  - `typescript` 5.6.0 → 5.9.3
  - `@types/express` 4.17.21 → 4.17.25
- All dependency versions are now pinned (no caret ranges) for reproducible builds.

## [1.0.0] - 2026-04-26

### Added
- Initial release
- Key management tools: `list_keys`, `get_key`, `create_key`, `update_key`, `delete_keys`
- Translation tools: `set_translation`, `set_translation_state`, `mark_outdated`
- Review tools: `list_comments`, `add_comment`, `resolve_comment`
- Language tools: `list_languages`, `list_namespaces`
- Task management: `list_tasks`, `get_task`, `create_task`, `update_task`, `finish_task`, `close_task`, `reopen_task`
- `TOLGEE_PROJECT_ID` is required (no implicit default)
