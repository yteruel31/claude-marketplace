/**
 * Output formatters for Tolgee MCP tools
 */

import type { ResponseFormat } from "../constants.js";
import type {
  Language,
  UsedNamespace,
  KeyWithTranslations,
  KeysListResponse,
  TranslationsResponse,
  Comment,
  Task,
  TasksListResponse,
} from "../types.js";
import { MAX_OUTPUT_LENGTH } from "../constants.js";

function truncateOutput(output: string): string {
  if (output.length > MAX_OUTPUT_LENGTH) {
    return output.slice(0, MAX_OUTPUT_LENGTH) + "\n\n... (output truncated)";
  }
  return output;
}

// ==================== Languages ====================

export function formatLanguages(languages: Language[], format: ResponseFormat): string {
  if (format === "json") {
    return truncateOutput(JSON.stringify(languages, null, 2));
  }

  if (languages.length === 0) {
    return "No languages found in this project.";
  }

  let output = `## Project Languages (${languages.length})\n\n`;
  output += "| ID | Tag | Name | Original | Base |\n";
  output += "|-----|-----|------|----------|------|\n";

  for (const lang of languages) {
    const flag = lang.flagEmoji || "";
    output += `| ${lang.id} | ${lang.tag} | ${flag} ${lang.name} | ${lang.originalName} | ${lang.base ? "Yes" : "No"} |\n`;
  }

  return truncateOutput(output);
}

// ==================== Namespaces ====================

export function formatNamespaces(namespaces: UsedNamespace[], format: ResponseFormat): string {
  if (format === "json") {
    return truncateOutput(JSON.stringify(namespaces, null, 2));
  }

  if (namespaces.length === 0) {
    return "No namespaces found in this project.";
  }

  let output = `## Namespaces (${namespaces.length})\n\n`;
  output += "| Namespace | Key Count |\n";
  output += "|-----------|----------|\n";

  for (const ns of namespaces) {
    output += `| ${ns.name || "(default)"} | ${ns.keyCount} |\n`;
  }

  return truncateOutput(output);
}

// ==================== Keys ====================

export function formatKeysList(response: KeysListResponse, format: ResponseFormat): string {
  if (format === "json") {
    return truncateOutput(JSON.stringify(response, null, 2));
  }

  const keys = response._embedded?.keys || [];
  const page = response.page;

  if (keys.length === 0) {
    return "No keys found matching your criteria.";
  }

  let output = `## Translation Keys`;
  if (page) {
    output += ` (Page ${page.number + 1} of ${page.totalPages}, ${page.totalElements} total)`;
  }
  output += "\n\n";

  for (const key of keys) {
    output += `### \`${key.name}\`\n`;
    output += `- **ID**: ${key.id}\n`;
    if (key.namespace) {
      output += `- **Namespace**: ${key.namespace}\n`;
    }
    if (key.tags && key.tags.length > 0) {
      output += `- **Tags**: ${key.tags.map((t) => t.name).join(", ")}\n`;
    }
    if (key.description) {
      output += `- **Description**: ${key.description}\n`;
    }

    // Translations
    if (key.translations && Object.keys(key.translations).length > 0) {
      output += "\n**Translations:**\n";
      for (const [lang, trans] of Object.entries(key.translations)) {
        const stateIcon = trans.state === "REVIEWED" ? "✓" : trans.state === "TRANSLATED" ? "○" : "·";
        const outdatedMark = trans.outdated ? " (outdated)" : "";
        output += `- \`${lang}\`: ${stateIcon} ${trans.text || "(empty)"}${outdatedMark}\n`;
      }
    }
    output += "\n";
  }

  return truncateOutput(output);
}

export function formatKey(key: KeyWithTranslations, format: ResponseFormat): string {
  if (format === "json") {
    return truncateOutput(JSON.stringify(key, null, 2));
  }

  let output = `## Key: \`${key.name}\`\n\n`;
  output += `- **ID**: ${key.id}\n`;
  if (key.namespace) {
    output += `- **Namespace**: ${key.namespace}\n`;
  }
  if (key.description) {
    output += `- **Description**: ${key.description}\n`;
  }
  if (key.tags && key.tags.length > 0) {
    output += `- **Tags**: ${key.tags.map((t) => t.name).join(", ")}\n`;
  }
  output += `- **Plural**: ${key.isPlural ? "Yes" : "No"}\n`;

  // Translations
  if (key.translations && Object.keys(key.translations).length > 0) {
    output += "\n### Translations\n\n";
    output += "| Language | State | Text | Outdated | Translation ID |\n";
    output += "|----------|-------|------|----------|----------------|\n";
    for (const [lang, trans] of Object.entries(key.translations)) {
      const text = trans.text || "(empty)";
      const displayText = text.length > 50 ? text.slice(0, 50) + "..." : text;
      output += `| ${lang} | ${trans.state} | ${displayText} | ${trans.outdated ? "Yes" : "No"} | ${trans.id} |\n`;
    }
  }

  // Screenshots
  if (key.screenshots && key.screenshots.length > 0) {
    output += `\n### Screenshots (${key.screenshots.length})\n`;
    for (const ss of key.screenshots) {
      output += `- ${ss.filename} (${ss.width}x${ss.height})\n`;
    }
  }

  return truncateOutput(output);
}

export function formatTranslationsResponse(response: TranslationsResponse, format: ResponseFormat): string {
  if (format === "json") {
    return truncateOutput(JSON.stringify(response, null, 2));
  }

  let output = `## Key: \`${response.keyName}\`\n\n`;
  output += `- **Key ID**: ${response.keyId}\n`;
  if (response.keyNamespace) {
    output += `- **Namespace**: ${response.keyNamespace}\n`;
  }

  if (response.translations && Object.keys(response.translations).length > 0) {
    output += "\n### Translations\n\n";
    for (const [lang, trans] of Object.entries(response.translations)) {
      output += `- **${lang}** (ID: ${trans.id}): ${trans.text || "(empty)"} [${trans.state}]\n`;
    }
  }

  return truncateOutput(output);
}

// ==================== Comments ====================

export function formatComments(comments: Comment[], translationId: number, format: ResponseFormat): string {
  if (format === "json") {
    return truncateOutput(JSON.stringify(comments, null, 2));
  }

  if (comments.length === 0) {
    return `No comments found for translation ID ${translationId}.`;
  }

  let output = `## Comments for Translation ${translationId} (${comments.length})\n\n`;

  for (const comment of comments) {
    const stateIcon =
      comment.state === "RESOLVED" ? "✓" : comment.state === "NEEDS_RESOLUTION" ? "!" : "·";
    output += `### ${stateIcon} Comment #${comment.id}\n`;
    output += `- **Author**: ${comment.author.name || comment.author.username}\n`;
    output += `- **State**: ${comment.state}\n`;
    output += `- **Created**: ${comment.createdAt}\n`;
    if (comment.updatedAt !== comment.createdAt) {
      output += `- **Updated**: ${comment.updatedAt}\n`;
    }
    output += `\n${comment.text}\n\n---\n\n`;
  }

  return truncateOutput(output);
}

export function formatComment(comment: Comment, format: ResponseFormat): string {
  if (format === "json") {
    return truncateOutput(JSON.stringify(comment, null, 2));
  }

  let output = `## Comment #${comment.id}\n\n`;
  output += `- **Author**: ${comment.author.name || comment.author.username}\n`;
  output += `- **State**: ${comment.state}\n`;
  output += `- **Created**: ${comment.createdAt}\n`;
  output += `\n${comment.text}\n`;

  return truncateOutput(output);
}

// ==================== Tasks ====================

export function formatTask(task: Task, format: ResponseFormat): string {
  if (format === "json") {
    return truncateOutput(JSON.stringify(task, null, 2));
  }

  const progress =
    task.totalItems > 0
      ? `${task.doneItems}/${task.totalItems} (${Math.round((task.doneItems / task.totalItems) * 100)}%)`
      : "0/0 (0%)";

  let output = `## Task #${task.number}${task.name ? `: ${task.name}` : ""}\n\n`;
  output += "| Property | Value |\n";
  output += "|----------|-------|\n";
  output += `| Type | ${task.type} |\n`;
  output += `| State | ${task.state} |\n`;
  output += `| Language | ${task.language.name} (${task.language.tag}) |\n`;
  output += `| Progress | ${progress} |\n`;
  output += `| Due Date | ${task.dueDate ? new Date(task.dueDate).toISOString().split("T")[0] : "None"} |\n`;
  output += `| Assignees | ${task.assignees.map((u) => u.name || u.username).join(", ") || "None"} |\n`;
  output += `| Word Count | ${task.baseWordCount} |\n`;
  output += `| Character Count | ${task.baseCharacterCount} |\n`;
  output += `| Author | ${task.author ? task.author.name || task.author.username : "Unknown"} |\n`;
  output += `| Created | ${new Date(task.createdAt).toISOString().split("T")[0]} |\n`;
  if (task.closedAt) {
    output += `| Closed | ${new Date(task.closedAt).toISOString().split("T")[0]} |\n`;
  }

  if (task.description) {
    output += `\n**Description:**\n${task.description}\n`;
  }

  return truncateOutput(output);
}

export function formatTaskList(response: TasksListResponse, format: ResponseFormat): string {
  if (format === "json") {
    return truncateOutput(JSON.stringify(response, null, 2));
  }

  const tasks = response._embedded?.tasks || [];
  const page = response.page;

  if (tasks.length === 0) {
    return "No tasks found matching your criteria.";
  }

  let output = `## Tasks`;
  if (page) {
    output += ` (Page ${page.number + 1} of ${page.totalPages}, ${page.totalElements} total)`;
  }
  output += "\n\n";

  output += "| # | Name | Type | State | Language | Progress |\n";
  output += "|---|------|------|-------|----------|----------|\n";

  for (const task of tasks) {
    const progress =
      task.totalItems > 0 ? `${Math.round((task.doneItems / task.totalItems) * 100)}%` : "0%";
    const name = task.name || "(unnamed)";
    const displayName = name.length > 30 ? name.slice(0, 30) + "..." : name;
    output += `| ${task.number} | ${displayName} | ${task.type} | ${task.state} | ${task.language.tag} | ${progress} |\n`;
  }

  return truncateOutput(output);
}
