/**
 * API Client v2 for Interview Practice Application
 * 
 * Clean, type-safe API client for backend v2 endpoints.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ==================== Types ====================

export interface CreateSessionRequest {
  job_description: string;
  user_background: string;
  interview_type: "Technical" | "Behavioral" | "Case Study";
  difficulty: "Beginner" | "Intermediate" | "Advanced";
}

export interface SendMessageRequest {
  message: string;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

export interface MessageResponse {
  session_id: string;
  ai_message: string;
  question_count: number;
  is_complete: boolean;
  timestamp: string;
}

export interface SessionResponse {
  session_id: string;
  interview_type: string;
  difficulty: string;
  question_count: number;
  is_active: boolean;
  is_complete: boolean;
  messages: Message[];
  created_at: string;
}

// ==================== API Client ====================

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    method: string,
    endpoint: string,
    data?: unknown
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const options: RequestInit = {
      method,
      headers: {
        "Content-Type": "application/json",
      },
    };

    if (data) {
      options.body = JSON.stringify(data);
    }

    const response = await fetch(url, options);

    if (!response.ok) {
      let errorMessage = `API Error ${response.status}`;
      
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          // FastAPI validation error format
          if (Array.isArray(errorData.detail)) {
            errorMessage = errorData.detail.map((err: any) => err.msg).join(", ");
          } else {
            errorMessage = errorData.detail;
          }
        }
      } catch {
        // If JSON parsing fails, try text
        const errorText = await response.text();
        if (errorText) errorMessage = errorText;
      }
      
      throw new Error(errorMessage);
    }

    // Handle 204 No Content
    if (response.status === 204) {
      return {} as T;
    }

    return response.json();
  }

  // ==================== Session Endpoints ====================

  async createSession(request: CreateSessionRequest): Promise<MessageResponse> {
    return this.request<MessageResponse>("POST", "/api/sessions", request);
  }

  async sendMessage(
    sessionId: string,
    request: SendMessageRequest
  ): Promise<MessageResponse> {
    return this.request<MessageResponse>(
      "POST",
      `/api/sessions/${sessionId}/message`,
      request
    );
  }

  async getSession(sessionId: string): Promise<SessionResponse> {
    return this.request<SessionResponse>("GET", `/api/sessions/${sessionId}`);
  }

  async deleteSession(sessionId: string): Promise<void> {
    return this.request<void>("DELETE", `/api/sessions/${sessionId}`);
  }

  // ==================== Health Check ====================

  async healthCheck(): Promise<{ status: string; version: string }> {
    return this.request("GET", "/health");
  }
}

// ==================== Export Singleton ====================

export const api = new ApiClient(API_BASE_URL);
