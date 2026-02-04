# Discord Plugin

MCP server for Discord bot integration and API documentation access.

## Features

### Bot Integration (requires DISCORD_BOT_TOKEN)
- **Send messages** to any accessible Discord channel
- **Read message history** with configurable limits
- **Search messages** by content
- **List guilds and channels** the bot has access to
- **Get detailed info** about channels and servers

### API Documentation (no auth required)
- **List documentation** sections and files
- **Search documentation** by keyword
- **Fetch documentation** pages directly

## Installation

The plugin is automatically loaded when added to your Claude marketplace.

## Requirements

### For Bot Tools
- **DISCORD_BOT_TOKEN** - Discord bot token from Developer Portal
- **uv** - Python package manager (https://github.com/astral-sh/uv)

### For Documentation Tools
- No requirements - works out of the box

## Bot Setup

1. Go to https://discord.com/developers/applications
2. Create a new application
3. Go to "Bot" section and click "Add Bot"
4. Enable Privileged Gateway Intents:
   - MESSAGE CONTENT INTENT
   - SERVER MEMBERS INTENT
5. Copy the bot token
6. Generate invite URL (OAuth2 > URL Generator):
   - Scopes: `bot`, `applications.commands`
   - Bot Permissions:
     - Read Messages/View Channels
     - Send Messages
     - Read Message History
7. Use the generated URL to invite the bot to your server

Set the environment variable:
```bash
export DISCORD_BOT_TOKEN="your-bot-token-here"
```

## Available Tools

### Bot Tools

| Tool | Description |
|------|-------------|
| `send_message` | Send a message to a channel |
| `read_messages` | Read recent messages from a channel |
| `search_messages` | Search messages by content |
| `list_guilds` | List all servers the bot is in |
| `list_channels` | List channels in a server |
| `get_channel_info` | Get detailed channel information |
| `get_guild_info` | Get detailed server information |

### Documentation Tools

| Tool | Description |
|------|-------------|
| `docs_list` | List available documentation sections |
| `docs_search` | Search documentation by keyword |
| `docs_fetch` | Fetch a specific documentation page |

## Examples

### Send a message
```
Send "Hello team!" to Discord channel 123456789012345678
```

### Read recent messages
```
Read the last 10 messages from Discord channel 123456789012345678
```

### Search documentation
```
Search Discord docs for "slash commands"
```

### Fetch documentation
```
Fetch the Discord API docs for interactions/application_commands
```

## Documentation Sections

The following Discord API documentation sections are available:

- `activities` - User activity features
- `change-log` - API version history
- `components` - UI components
- `developer-tools` - Development tools
- `discord-social-sdk` - Social SDK
- `discovery` - Discovery features
- `events` - Gateway events
- `interactions` - Slash commands, buttons, etc.
- `monetization` - Monetization features
- `policies-and-agreements` - Legal docs
- `quick-start` - Getting started guides
- `resources` - API resources (channels, guilds, etc.)
- `rich-presence` - Rich presence
- `topics` - General topics
- `tutorials` - Step-by-step guides

## Security Considerations

- Never share your bot token
- Use minimal required permissions
- The bot can only access channels with proper permissions
- Consider using separate bots for different purposes

## Configuration

The MCP server is configured in `.mcp.json`:

```json
{
  "mcpServers": {
    "discord": {
      "command": "bash",
      "args": ["${CLAUDE_PLUGIN_ROOT}/servers/discord-mcp/run.sh"],
      "env": {
        "DISCORD_BOT_TOKEN": "${DISCORD_BOT_TOKEN}"
      }
    }
  }
}
```
