---
name: tolgee
description: Manage translations in Tolgee via MCP. Use when user asks to update translations, create or search keys, review copy, set translation states, or manage translation tasks. Relevant for i18n, localization, copy changes, or any Tolgee operations.
---

# Tolgee MCP Server Guide

This skill enables programmatic translation management in Tolgee. Use these tools to search, create, update, and review translations without leaving your development environment.

## Quick Start

1. **List available languages**: `list_languages`
2. **Search for a key**: `list_keys` with search term, filters, or sorting
3. **Update a translation**: `set_translation`
4. **Mark as reviewed**: `set_translation_state`

## Tool Reference

### Key Management

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `list_keys` | Search and list translation keys | `search`, `namespace`, `tags`, `filter_state`, `languages`, `filter_untranslated_in_lang`, `filter_translated_in_lang`, `filter_has_screenshot`, `filter_key_name`, `sort`, `page`, `size` |
| `get_key` | Get key details with all translations | `key_id` |
| `create_key` | Create new translation key | `key_name`, `translations`, `namespace`, `tags` |
| `update_key` | Update key and translations | `key_id`, `name`, `translations`, `tags` |
| `delete_keys` | Delete keys (destructive) | `key_ids` |

### Translation Management

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `set_translation` | Set/update translation text | `key_name`, `translations` |
| `set_translation_state` | Change state (TRANSLATED/REVIEWED) | `translation_id`, `state` |
| `mark_outdated` | Flag translation needs update | `translation_id`, `outdated` |

### Review & Comments

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `list_comments` | List comments on translation | `translation_id` |
| `add_comment` | Add review comment | `translation_id`, `text`, `state` |
| `resolve_comment` | Mark comment resolved | `translation_id`, `comment_id` |

### Organization

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `list_languages` | List project languages | - |
| `list_namespaces` | List namespaces with key counts | - |

### Task Management

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `list_tasks` | List tasks with filtering | `state`, `type`, `assignee_id`, `page`, `size` |
| `get_task` | Get task details | `task_number` |
| `create_task` | Create translation/review task | `type`, `language_id`, `key_ids`, `name`, `description` |
| `update_task` | Update task details | `task_number`, `name`, `description`, `due_date` |
| `finish_task` | Mark task as done | `task_number` |
| `close_task` | Close/archive task | `task_number` |
| `reopen_task` | Reopen closed task | `task_number` |

## Common Workflows

### 1. Find and Update a Translation

```
# Search for the key
list_keys(search="checkout.button")

# Get full details (note the key ID and translation IDs)
get_key(key_id=12345)

# Update the English translation
set_translation(
  key_name="checkout.pay_button",
  translations={"en": "Complete Payment", "sv": "Slutför betalning"}
)
```

### 2. Add a New Translation Key

```
create_key(
  key_name="settings.notifications.email_toggle",
  namespace="settings",
  translations={
    "en": "Email notifications",
    "sv": "E-postnotifikationer"
  },
  tags=["settings", "notifications"]
)
```

### 3. Review Translation Workflow

```
# Find keys needing review
list_keys(namespace="checkout")

# Get key details to see translation IDs
get_key(key_id=12345)

# Mark translation as reviewed (use translation_id, not key_id)
set_translation_state(translation_id=67890, state="REVIEWED")
```

### 4. Add Review Feedback

```
# Add a comment requesting changes
add_comment(
  translation_id=67890,
  text="Should this use formal or informal tone?",
  state="NEEDS_RESOLUTION"
)

# Later, resolve the comment
resolve_comment(translation_id=67890, comment_id=111)
```

### 5. Bulk Update for a Feature

```
# List all keys in a namespace
list_keys(namespace="onboarding", size=100)

# Update multiple translations
set_translation(
  key_name="onboarding.welcome_title",
  translations={"en": "Welcome!", "sv": "Välkommen!"}
)
```

### 6. Advanced Filtering

```
# Find all untranslated keys in French
list_keys(
  filter_state=["UNTRANSLATED"],
  filter_untranslated_in_lang=["fr"]
)

# List keys with translations in English and Swedish, sorted by name
list_keys(
  languages=["en", "sv"],
  sort=["key,asc"],
  size=50
)

# Find keys matching a name pattern that have screenshots
list_keys(
  filter_key_name="checkout",
  filter_has_screenshot=true
)
```

### 7. Create and Manage Translation Tasks

```
# Create a translation task
create_task(
  type="TRANSLATE",
  language_id=1360137002,
  key_ids=[12345, 12346, 12347],
  name="Translate checkout flow",
  description="Priority: High"
)

# Check progress
get_task(task_number=66)

# Mark as done
finish_task(task_number=66)
```

## Key Naming Conventions

Use dot notation for hierarchical organization:
- `{feature}.{component}.{element}` — e.g., `checkout.payment.submit_button`
- `{page}.{section}.{text}` — e.g., `home.hero.headline`

Common prefixes:
- `common.` — shared across the app (buttons, labels)
- `error.` — error messages
- `validation.` — form validation messages
- `{feature}.` — feature-specific translations

## Translation States

| State | Description |
|-------|-------------|
| `UNTRANSLATED` | No translation yet |
| `TRANSLATED` | Translation provided |
| `REVIEWED` | Translation approved |

## Task States

| State | Description |
|-------|-------------|
| `NEW` | Task created, not started |
| `IN_PROGRESS` | Work has begun |
| `DONE` | Work completed (`finish_task`) |
| `CLOSED` | Task archived/cancelled (`close_task`) |

## Comment States

| State | Description |
|-------|-------------|
| `RESOLUTION_NOT_NEEDED` | Informational comment |
| `NEEDS_RESOLUTION` | Action required |
| `RESOLVED` | Issue addressed |

## Response Formats

All tools support `response_format`:
- `markdown` (default) — human-readable formatted output
- `json` — raw API response for programmatic use

## Important Notes

1. **Translation IDs vs Key IDs**: `set_translation_state` and comment tools use `translation_id` (the ID of a specific language translation), not `key_id`. Use `get_key` to find translation IDs.

2. **Key Creation**: Use `create_key` with initial translations or `set_translation` which creates the key if it doesn't exist.

3. **Destructive Operations**: `delete_keys` permanently deletes keys and all their translations. There is no undo.

4. **Namespaces**: Organize keys by feature/domain. Default namespace is `null`.

5. **Tags**: Use tags for cross-cutting concerns (e.g., `needs-review`, `legal`, `marketing`).

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| Authentication failed | Invalid API key | Check `TOLGEE_API_KEY` |
| Permission denied | API key lacks scope | Request key with required permissions |
| Resource not found | Invalid key/translation ID | Verify ID with list/get tools |
| Rate limited | Too many requests | Wait and retry |

## Environment Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TOLGEE_API_KEY` | Yes | - | Project API key (`tgpak_...`) |
| `TOLGEE_PROJECT_ID` | Yes | - | Tolgee project ID |
| `TOLGEE_HOST` | No | `https://app.tolgee.io` | Tolgee instance URL |
