# Tolgee Plugin

MCP server for managing translations in Tolgee programmatically. Enables searching, creating, updating, and reviewing translation keys directly from Claude Code.

## Features

- **Key Management**: Search, create, update, and delete translation keys
- **Translation Management**: Set translations, change states, mark as outdated
- **Review Workflow**: Add and resolve comments for collaboration
- **Task Management**: Create, update, finish, close, and reopen translation/review tasks
- **Organization**: Browse languages and namespaces

## Quick Start

1. Set your Tolgee API key and project ID:
   ```bash
   export TOLGEE_API_KEY="tgpak_your_key_here"
   export TOLGEE_PROJECT_ID="12345"
   ```

2. Optionally configure a self-hosted Tolgee instance:
   ```bash
   export TOLGEE_HOST="https://your-tolgee.example.com"
   ```

## Tools

| Tool | Description |
|------|-------------|
| `list_keys` | Search and list translation keys with advanced filtering (state, languages, screenshots, sorting) |
| `get_key` | Get key details with all translations |
| `create_key` | Create new translation key |
| `update_key` | Update key and translations |
| `delete_keys` | Delete keys |
| `set_translation` | Set/update translation text |
| `set_translation_state` | Change state (TRANSLATED/REVIEWED) |
| `mark_outdated` | Flag translation needs update |
| `list_comments` | List comments on translation |
| `add_comment` | Add review comment |
| `resolve_comment` | Mark comment resolved |
| `list_languages` | List project languages |
| `list_namespaces` | List namespaces |
| `list_tasks` | List tasks with filtering |
| `get_task` | Get task details |
| `create_task` | Create translation/review task |
| `update_task` | Update task details |
| `finish_task` | Mark task as done |
| `close_task` | Close/archive task |
| `reopen_task` | Reopen closed task |

## Example Usage

```
# Find a translation
list_keys(search="checkout")

# Find untranslated keys in French
list_keys(filter_untranslated_in_lang=["fr"], filter_state=["UNTRANSLATED"])

# List keys with specific languages, sorted by name
list_keys(languages=["en", "sv"], sort=["key,asc"])

# Update translation
set_translation(
  key_name="checkout.pay_button",
  translations={"en": "Pay Now", "sv": "Betala nu"}
)

# Mark as reviewed
set_translation_state(translation_id=12345, state="REVIEWED")
```

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TOLGEE_API_KEY` | Yes | - | Project API key (format: `tgpak_...`) |
| `TOLGEE_PROJECT_ID` | Yes | - | Project ID |
| `TOLGEE_HOST` | No | `https://app.tolgee.io` | Tolgee instance URL |

## Requirements

- Node.js >= 18.0.0
- A Tolgee account with API access (cloud or self-hosted)

## Documentation

See [skills/tolgee/SKILL.md](skills/tolgee/SKILL.md) for the full usage guide and workflow examples.
