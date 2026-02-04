---
name: discord
description: "Discord bot integration and API documentation. Send messages, read channels, list servers, and fetch Discord API docs."
triggers:
  - discord
  - send discord message
  - post to discord
  - read discord
  - discord channel
  - discord server
  - list discord servers
  - list discord channels
  - discord history
  - discord messages
  - check discord
  - message on discord
  - discord api docs
  - discord documentation
  - discord api reference
  - how does discord
  - discord developer docs
---

# Discord Skill

This skill provides Discord bot integration and API documentation access.

## Available Tools

### Bot Tools (require DISCORD_BOT_TOKEN)

#### `send_message`

Send a message to a Discord channel.

**Parameters:**
- `channel_id` (required): Discord channel ID
- `content` (required): Message content (max 2000 characters)
- `reply_to_message_id` (optional): Message ID to reply to

**Example:**
```
Use send_message with channel_id: "123456789" and content: "Hello from Claude!"
```

#### `read_messages`

Read recent messages from a Discord channel.

**Parameters:**
- `channel_id` (required): Discord channel ID
- `limit` (optional): Number of messages (1-100, default 25)

**Example:**
```
Use read_messages with channel_id: "123456789" and limit: 10
```

#### `search_messages`

Search for messages containing specific text.

**Parameters:**
- `channel_id` (required): Channel to search in
- `query` (required): Search text (case-insensitive)
- `limit` (optional): Max messages to search (1-100, default 50)

**Example:**
```
Use search_messages with channel_id: "123456789" and query: "deployment"
```

#### `list_guilds`

List all Discord servers the bot is a member of.

**Parameters:** None

**Example:**
```
Use list_guilds to see all connected servers
```

#### `list_channels`

List all channels in a Discord server.

**Parameters:**
- `guild_id` (required): Discord server ID
- `channel_type` (optional): Filter by "text", "voice", "category", or "all"

**Example:**
```
Use list_channels with guild_id: "123456789" and channel_type: "text"
```

#### `get_channel_info`

Get detailed information about a specific channel.

**Parameters:**
- `channel_id` (required): Discord channel ID

**Example:**
```
Use get_channel_info with channel_id: "123456789"
```

#### `get_guild_info`

Get detailed information about a Discord server.

**Parameters:**
- `guild_id` (required): Discord server ID

**Example:**
```
Use get_guild_info with guild_id: "123456789"
```

### Documentation Tools (no auth required)

#### `docs_list`

List available Discord API documentation sections and files.

**Parameters:**
- `section` (optional): Filter by section (e.g., "interactions", "resources")

**Available sections:**
- activities, change-log, components, developer-tools
- discord-social-sdk, discovery, events, interactions
- monetization, policies-and-agreements, quick-start
- resources, rich-presence, topics, tutorials

**Example:**
```
Use docs_list to see all available documentation
Use docs_list with section: "interactions" to see interaction-related docs
```

#### `docs_search`

Search Discord API documentation by keyword.

**Parameters:**
- `query` (required): Search query (min 2 chars)
- `limit` (optional): Max results (1-50, default 10)

**Example:**
```
Use docs_search with query: "slash commands"
Use docs_search with query: "gateway" and limit: 20
```

#### `docs_fetch`

Fetch the content of a specific documentation page.

**Parameters:**
- `path` (required): Path to doc file (e.g., "interactions/application_commands.mdx")

**Example:**
```
Use docs_fetch with path: "interactions/application_commands.mdx"
Use docs_fetch with path: "resources/channel.mdx"
```

## Common Use Cases

### Send a notification
```
Send a message to Discord channel 123456789 saying "Build completed!"
```

### Check recent activity
```
Read the last 20 messages from Discord channel 123456789
```

### Find specific discussions
```
Search for messages about "deployment" in Discord channel 123456789
```

### Get server overview
```
List all Discord servers the bot is in
```

### Look up API documentation
```
Search the Discord docs for "slash commands"
Fetch the Discord API docs for interactions/application_commands
```

## Configuration

### Bot Tools
Requires `DISCORD_BOT_TOKEN` environment variable.

**Bot Setup:**
1. Go to https://discord.com/developers/applications
2. Create a new application
3. Go to "Bot" section and create a bot
4. Enable Privileged Gateway Intents:
   - MESSAGE CONTENT INTENT
   - SERVER MEMBERS INTENT
5. Copy the bot token
6. OAuth2 > URL Generator:
   - Scopes: `bot`, `applications.commands`
   - Permissions: Read/Send Messages, Read Message History
7. Invite bot to your server

```bash
export DISCORD_BOT_TOKEN="your-bot-token-here"
```

### Documentation Tools
No configuration needed. Works out of the box.

## Limitations

- Message content is limited to 2000 characters
- Bot can only access channels it has permission to view
- Rate limits apply (handled automatically)
- Message search is local (fetches then filters, not Discord's search API)
- Documentation search limited to first 50 files for content matching
