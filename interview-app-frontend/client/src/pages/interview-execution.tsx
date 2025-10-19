import { useEffect, useState, useRef } from "react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ArrowLeft, Send, Loader2, Sparkles, User, Clock } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, type Message } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

export default function InterviewExecution() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [userMessage, setUserMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [interviewConfig, setInterviewConfig] = useState<any>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Fetch session with all messages
  const { data: session, refetch } = useQuery({
    queryKey: ["session", sessionId],
    queryFn: () => api.getSession(sessionId!),
    enabled: !!sessionId,
    refetchInterval: false,
  });

  const messages = session?.messages || [];
  const isInterviewComplete = session?.is_complete || false;

  const createSessionMutation = useMutation({
    mutationFn: async (config: any) => {
      const payload = {
        job_description: config.jobDescription || "",
        user_background: config.candidateBackground || "",
        interview_type: config.interviewType === "case_study" ? "Case Study" : 
                       config.interviewType === "behavioral" ? "Behavioral" : "Technical",
        difficulty: config.difficulty.charAt(0).toUpperCase() + config.difficulty.slice(1) as any,
      };
      
      return api.createSession(payload);
    },
    onSuccess: (data) => {
      setSessionId(data.session_id);
    },
    onError: (error: Error) => {
      toast({
        title: "Error creating session",
        description: error.message || "Please try again",
        variant: "destructive",
      });
    },
  });

  const sendMessageMutation = useMutation({
    mutationFn: async ({ sessionId, message }: { sessionId: string; message: string }) => {
      return api.sendMessage(sessionId, { message });
    },
    onMutate: () => {
      setIsTyping(true);
    },
    onSuccess: async () => {
      setUserMessage("");
      await refetch(); // Wait for refetch to complete
      setTimeout(() => setIsTyping(false), 500);
    },
    onError: (error: Error) => {
      setIsTyping(false);
      toast({
        title: "Error sending message",
        description: error.message || "Please try again",
        variant: "destructive",
      });
    },
  });

  useEffect(() => {
    const stored = localStorage.getItem("interview-config");
    if (!stored) {
      setLocation("/");
      return;
    }
    const config = JSON.parse(stored);
    setInterviewConfig(config);
    createSessionMutation.mutate(config);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = () => {
    if (!sessionId || !userMessage.trim() || sendMessageMutation.isPending) return;
    
    // Validate message length
    if (userMessage.length > 5000) {
      toast({
        title: "Message too long",
        description: "Please keep your answer under 5000 characters",
        variant: "destructive",
      });
      return;
    }
    
    sendMessageMutation.mutate({ sessionId, message: userMessage.trim() });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!interviewConfig || createSessionMutation.isPending) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <p className="text-sm text-muted-foreground">Setting up your interview...</p>
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col">
      {/* Header */}
      <div className="border-b bg-card/50 px-4 py-3">
        <div className="container mx-auto flex items-center justify-between">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setLocation("/configure")}
            data-testid="button-exit"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Exit Interview
          </Button>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm">
              <Badge variant="outline" data-testid="badge-interview-type">
                {interviewConfig.interviewType.replace("_", " ")}
              </Badge>
              <Badge variant="outline" data-testid="badge-difficulty-level">
                {interviewConfig.difficulty}
              </Badge>
            </div>
            <Separator orientation="vertical" className="h-6" />
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Clock className="h-4 w-4" />
              <span data-testid="text-message-count">Question {session?.question_count || 0}/6</span>
            </div>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto">
        <div className="container mx-auto max-w-4xl px-4 py-6">
          <div className="space-y-6">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex gap-4 ${message.role === "user" ? "justify-end" : ""}`}
                data-testid={`message-${message.role}-${index}`}
              >
                {message.role === "assistant" && (
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary to-primary/70">
                    <Sparkles className="h-5 w-5 text-primary-foreground" />
                  </div>
                )}
                
                <div
                  className={`max-w-[80%] space-y-2 ${
                    message.role === "assistant"
                      ? "rounded-2xl rounded-tl-sm bg-card p-4 md:p-6"
                      : "rounded-2xl rounded-tr-sm bg-primary/10 p-4 md:p-6"
                  }`}
                >
                  <p className="whitespace-pre-wrap leading-relaxed">
                    {message.content}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {new Date(message.timestamp).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                </div>

                {message.role === "user" && (
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-muted">
                    <User className="h-5 w-5 text-muted-foreground" />
                  </div>
                )}
              </div>
            ))}

            {isTyping && (
              <div className="flex gap-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary to-primary/70">
                  <Sparkles className="h-5 w-5 text-primary-foreground" />
                </div>
                <div className="rounded-2xl rounded-tl-sm bg-card p-4">
                  <div className="flex gap-1">
                    <div className="h-2 w-2 animate-pulse rounded-full bg-primary" />
                    <div className="h-2 w-2 animate-pulse rounded-full bg-primary delay-150" />
                    <div className="h-2 w-2 animate-pulse rounded-full bg-primary delay-300" />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t bg-background">
        <div className="container mx-auto max-w-4xl p-4">
          {isInterviewComplete && (
            <div className="mb-4 rounded-lg border border-primary/20 bg-primary/5 p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                    <Sparkles className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <h4 className="font-medium text-sm">Interview Complete!</h4>
                    <p className="text-xs text-muted-foreground">Great job! Ready to practice again?</p>
                  </div>
                </div>
                <Button
                  onClick={() => setLocation("/")}
                  variant="default"
                  size="sm"
                >
                  Start New Interview
                </Button>
              </div>
            </div>
          )}
          <div className="space-y-2">
            <div className="flex gap-3">
              <div className="flex-1 space-y-1">
                <Textarea
                  value={userMessage}
                  onChange={(e) => setUserMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={isInterviewComplete ? "Interview complete! Start a new session to practice again." : "Type your answer... (Press Enter to send, Shift+Enter for new line)"}
                  className="min-h-12 max-h-32 resize-none"
                  disabled={isInterviewComplete}
                  data-testid="input-message"
                />
                <p className={`text-xs text-right ${userMessage.length > 4500 ? 'text-destructive' : 'text-muted-foreground'}`}>
                  {userMessage.length} / 5000 characters
                </p>
              </div>
              <Button
                size="icon"
                onClick={handleSendMessage}
                disabled={!userMessage.trim() || sendMessageMutation.isPending || isInterviewComplete}
                className="h-12 w-12 shrink-0"
                data-testid="button-send"
              >
                <Send className="h-5 w-5" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
