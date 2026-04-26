# Changelog

All notable changes to the tolgee plugin will be documented in this file.

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
