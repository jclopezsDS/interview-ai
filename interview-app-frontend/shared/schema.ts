// Interview context type for parsing (used by the UI for typing requests)
export type InterviewContext = {
  jobDescription: string;
  candidateBackground: string;
  companyInfo?: string;
  roleDetails?: string;
};

// Parsed context type returned from AI (as consumed by the UI)
export type ParsedContext = {
  jobTitle?: string;
  requiredSkills?: string[];
  companyName?: string;
  yearsOfExperience?: number;
  suggestedInterviewType?: string;
  suggestedDifficulty?: string;
};

// Interview configuration
export type InterviewConfig = {
  interviewType: "technical" | "behavioral" | "case_study";
  difficulty: "beginner" | "intermediate" | "advanced";
};

// Chat message type used in the UI conversation views
export type ChatMessage = {
  id?: string;
  sessionId: string;
  role: "ai" | "user";
  content: string;
  timestamp?: string | Date;
};

// Optional session type for UI state (kept lightweight for frontend-only)
export type InterviewSession = {
  id?: string;
  jobDescription: string;
  candidateBackground: string;
  companyInfo?: string;
  roleDetails?: string;
  interviewType: "technical" | "behavioral" | "case_study";
  difficulty: "beginner" | "intermediate" | "advanced";
  status?: "in_progress" | "completed";
  createdAt?: string | Date;
};
