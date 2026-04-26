/**
 * Tolgee API client - singleton pattern
 */

import axios, { AxiosInstance, AxiosError } from "axios";
import {
  DEFAULT_HOST,
  API_TIMEOUT,
  DEFAULT_PAGE_SIZE,
  type TranslationState,
  type FilterTranslationState,
  type CommentState,
  type TaskType,
  type TaskState,
} from "../constants.js";
import type {
  Language,
  UsedNamespace,
  KeyWithTranslations,
  KeysListResponse,
  TranslationsResponse,
  Comment,
  CreateKeyRequest,
  UpdateKeyRequest,
  SetTranslationRequest,
  CreateCommentRequest,
  Task,
  TasksListResponse,
  CreateTaskRequest,
  UpdateTaskRequest,
} from "../types.js";

/**
 * Check if a value is an unresolved MCP template variable
 * e.g., "${TOLGEE_HOST}" instead of "https://app.tolgee.io"
 */
function isUnresolvedTemplate(value: string | undefined): boolean {
  return typeof value === "string" && value.startsWith("${") && value.endsWith("}");
}

export class TolgeeClient {
  private client: AxiosInstance;
  private projectId: number;
  private host: string;

  constructor(host?: string, apiKey?: string, projectId?: number) {
    const key = apiKey || process.env.TOLGEE_API_KEY;
    if (!key || isUnresolvedTemplate(key)) {
      throw new Error(
        "Tolgee API key is required. Set TOLGEE_API_KEY environment variable or pass it to the constructor."
      );
    }

    // Handle TOLGEE_HOST with template detection
    const envHost = process.env.TOLGEE_HOST;
    const resolvedHost = envHost && !isUnresolvedTemplate(envHost) ? envHost : undefined;
    this.host = (host || resolvedHost || DEFAULT_HOST).replace(/\/$/, "");

    // Handle TOLGEE_PROJECT_ID with template detection (required, no default)
    if (projectId !== undefined) {
      this.projectId = projectId;
    } else {
      const envProjectId = process.env.TOLGEE_PROJECT_ID;
      if (!envProjectId || isUnresolvedTemplate(envProjectId)) {
        throw new Error(
          "Tolgee project ID is required. Set TOLGEE_PROJECT_ID environment variable or pass it to the constructor."
        );
      }
      const parsed = parseInt(envProjectId, 10);
      if (isNaN(parsed)) {
        throw new Error(`Invalid TOLGEE_PROJECT_ID: "${envProjectId}" is not a valid integer.`);
      }
      this.projectId = parsed;
    }

    this.client = axios.create({
      baseURL: `${this.host}/v2/projects/${this.projectId}`,
      headers: {
        "X-API-Key": key,
        "Content-Type": "application/json",
      },
      timeout: API_TIMEOUT,
    });
  }

  private handleError(error: unknown): never {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<{ message?: string; code?: string; params?: unknown[] }>;
      const status = axiosError.response?.status;
      const data = axiosError.response?.data;
      const message = data?.message || axiosError.message;
      const params = data?.params ? ` (params: ${JSON.stringify(data.params)})` : "";

      switch (status) {
        case 401:
          throw new Error(`Authentication failed: Invalid API key. ${message}`);
        case 403:
          throw new Error(`Permission denied: API key lacks required permissions. ${message}`);
        case 404:
          throw new Error(`Resource not found: ${message}`);
        case 429:
          throw new Error(`Rate limited: Too many requests. Please wait before retrying.`);
        case 400:
          throw new Error(`Bad request: ${message}${params}`);
        default:
          throw new Error(`Tolgee API error (${status || "unknown"}): ${message}${params}`);
      }
    }
    throw error;
  }

  // ==================== Languages ====================

  async listLanguages(): Promise<Language[]> {
    try {
      const response = await this.client.get<{ _embedded?: { languages?: Language[] } }>("/languages");
      return response.data._embedded?.languages || [];
    } catch (error) {
      this.handleError(error);
    }
  }

  // ==================== Namespaces ====================

  async listNamespaces(): Promise<UsedNamespace[]> {
    try {
      const response = await this.client.get<{ _embedded?: { namespaces?: UsedNamespace[] } }>("/used-namespaces");
      return response.data._embedded?.namespaces || [];
    } catch (error) {
      this.handleError(error);
    }
  }

  // ==================== Keys ====================

  async listKeys(params?: {
    namespace?: string;
    search?: string;
    tags?: string[];
    filterState?: FilterTranslationState[];
    languages?: string[];
    filterUntranslatedInLang?: string[];
    filterTranslatedInLang?: string[];
    filterHasScreenshot?: boolean;
    filterKeyName?: string;
    sort?: string[];
    page?: number;
    size?: number;
  }): Promise<KeysListResponse> {
    try {
      const queryParams: Record<string, string | number | boolean | string[]> = {
        size: params?.size || DEFAULT_PAGE_SIZE,
        page: params?.page || 0,
      };

      if (params?.namespace) {
        queryParams.filterNamespace = params.namespace;
      }
      if (params?.search) {
        queryParams.search = params.search;
      }
      if (params?.tags && params.tags.length > 0) {
        queryParams.filterTag = params.tags;
      }
      if (params?.filterState && params.filterState.length > 0) {
        queryParams.filterState = params.filterState;
      }
      if (params?.languages && params.languages.length > 0) {
        queryParams.languages = params.languages;
      }
      if (params?.filterUntranslatedInLang && params.filterUntranslatedInLang.length > 0) {
        queryParams.filterUntranslatedInLang = params.filterUntranslatedInLang;
      }
      if (params?.filterTranslatedInLang && params.filterTranslatedInLang.length > 0) {
        queryParams.filterTranslatedInLang = params.filterTranslatedInLang;
      }
      if (params?.filterHasScreenshot !== undefined) {
        queryParams.filterHasScreenshot = params.filterHasScreenshot;
      }
      if (params?.filterKeyName) {
        queryParams.filterKeyName = params.filterKeyName;
      }
      if (params?.sort && params.sort.length > 0) {
        queryParams.sort = params.sort;
      }

      const response = await this.client.get<KeysListResponse>("/translations", { params: queryParams });
      return response.data;
    } catch (error) {
      this.handleError(error);
    }
  }

  async getKey(keyId: number): Promise<KeyWithTranslations> {
    try {
      const response = await this.client.get<KeyWithTranslations>(`/keys/${keyId}/info`);
      return response.data;
    } catch (error) {
      this.handleError(error);
    }
  }

  async createKey(request: CreateKeyRequest): Promise<TranslationsResponse> {
    try {
      const response = await this.client.post<TranslationsResponse>("/translations", request);
      return response.data;
    } catch (error) {
      this.handleError(error);
    }
  }

  async updateKey(keyId: number, request: UpdateKeyRequest): Promise<KeyWithTranslations> {
    try {
      const response = await this.client.put<KeyWithTranslations>(`/keys/${keyId}/complex-update`, request);
      return response.data;
    } catch (error) {
      this.handleError(error);
    }
  }

  async deleteKeys(keyIds: number[]): Promise<void> {
    try {
      await this.client.delete("/keys", { data: { ids: keyIds } });
    } catch (error) {
      this.handleError(error);
    }
  }

  // ==================== Translations ====================

  async setTranslation(request: SetTranslationRequest): Promise<TranslationsResponse> {
    try {
      const response = await this.client.put<TranslationsResponse>("/translations", request);
      return response.data;
    } catch (error) {
      this.handleError(error);
    }
  }

  async setTranslationState(translationId: number, state: TranslationState): Promise<void> {
    try {
      await this.client.put(`/translations/${translationId}/set-state`, { state });
    } catch (error) {
      this.handleError(error);
    }
  }

  async setTranslationOutdated(translationId: number, outdated: boolean): Promise<void> {
    try {
      await this.client.put(`/translations/${translationId}/set-outdated`, { outdated });
    } catch (error) {
      this.handleError(error);
    }
  }

  // ==================== Comments ====================

  async listComments(translationId: number): Promise<Comment[]> {
    try {
      const response = await this.client.get<{ _embedded?: { comments?: Comment[] } }>(
        `/translations/${translationId}/comments`
      );
      return response.data._embedded?.comments || [];
    } catch (error) {
      this.handleError(error);
    }
  }

  async addComment(translationId: number, request: CreateCommentRequest): Promise<Comment> {
    try {
      const response = await this.client.post<Comment>(`/translations/${translationId}/comments`, request);
      return response.data;
    } catch (error) {
      this.handleError(error);
    }
  }

  async setCommentState(translationId: number, commentId: number, state: CommentState): Promise<void> {
    try {
      await this.client.put(`/translations/${translationId}/comments/${commentId}/set-state/${state}`);
    } catch (error) {
      this.handleError(error);
    }
  }

  // ==================== Tasks ====================

  async listTasks(params?: {
    state?: TaskState;
    type?: TaskType;
    assigneeId?: number;
    page?: number;
    size?: number;
  }): Promise<TasksListResponse> {
    try {
      const queryParams: Record<string, string | number> = {
        size: params?.size || DEFAULT_PAGE_SIZE,
        page: params?.page || 0,
      };
      if (params?.state) queryParams.filterState = params.state;
      if (params?.type) queryParams.filterType = params.type;
      if (params?.assigneeId) queryParams.filterAssignee = params.assigneeId;

      const response = await this.client.get<TasksListResponse>("/tasks", { params: queryParams });
      return response.data;
    } catch (error) {
      this.handleError(error);
    }
  }

  async getTask(taskNumber: number): Promise<Task> {
    try {
      const response = await this.client.get<Task>(`/tasks/${taskNumber}`);
      return response.data;
    } catch (error) {
      this.handleError(error);
    }
  }

  async createTask(request: CreateTaskRequest): Promise<Task> {
    try {
      const response = await this.client.post<Task>("/tasks", request);
      return response.data;
    } catch (error) {
      this.handleError(error);
    }
  }

  async updateTask(taskNumber: number, request: UpdateTaskRequest): Promise<Task> {
    try {
      const response = await this.client.put<Task>(`/tasks/${taskNumber}`, request);
      return response.data;
    } catch (error) {
      this.handleError(error);
    }
  }

  async finishTask(taskNumber: number): Promise<void> {
    try {
      await this.client.put(`/tasks/${taskNumber}/finish`);
    } catch (error) {
      this.handleError(error);
    }
  }

  async closeTask(taskNumber: number): Promise<void> {
    try {
      await this.client.put(`/tasks/${taskNumber}/close`);
    } catch (error) {
      this.handleError(error);
    }
  }

  async reopenTask(taskNumber: number): Promise<void> {
    try {
      await this.client.put(`/tasks/${taskNumber}/reopen`);
    } catch (error) {
      this.handleError(error);
    }
  }

  // ==================== Utility ====================

  getProjectId(): number {
    return this.projectId;
  }

  getHost(): string {
    return this.host;
  }
}

// Singleton instance
let clientInstance: TolgeeClient | null = null;

export function getTolgeeClient(host?: string, apiKey?: string, projectId?: number): TolgeeClient {
  if (!clientInstance || host || apiKey || projectId) {
    clientInstance = new TolgeeClient(host, apiKey, projectId);
  }
  return clientInstance;
}

export function resetTolgeeClient(): void {
  clientInstance = null;
}
