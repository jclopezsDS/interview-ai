import { useEffect, useState } from "react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Brain, MessageSquare, Briefcase, ArrowLeft, Loader2 } from "lucide-react";
import { ProgressStepper } from "@/components/progress-stepper";
import { useToast } from "@/hooks/use-toast";

const steps = [
  { id: 1, title: "Context", description: "Provide details" },
  { id: 2, title: "Configure", description: "Set preferences" },
  { id: 3, title: "Interview", description: "Practice live" },
];

const interviewTypes = [
  {
    id: "technical",
    title: "Technical",
    description: "Coding, system design, algorithms",
    icon: Brain,
  },
  {
    id: "behavioral",
    title: "Behavioral",
    description: "Past experiences, soft skills",
    icon: MessageSquare,
  },
  {
    id: "case_study",
    title: "Case Study",
    description: "Problem solving, business scenarios",
    icon: Briefcase,
  },
];

const difficultyLevels = [
  { value: 0, label: "Beginner" },
  { value: 50, label: "Intermediate" },
  { value: 100, label: "Advanced" },
];

export default function InterviewConfiguration() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [contextData, setContextData] = useState<any>(null);
  const [selectedType, setSelectedType] = useState<string>("technical");
  const [difficultyValue, setDifficultyValue] = useState<number[]>([50]);

  useEffect(() => {
    const stored = localStorage.getItem("interview-context");
    if (!stored) {
      setLocation("/");
      return;
    }
    const data = JSON.parse(stored);
    setContextData(data);
  }, []);

  const handleStartInterview = () => {
    const difficultyLabel = difficultyLevels.find((d) => d.value === difficultyValue[0])?.label.toLowerCase() || "intermediate";
    const config = {
      ...contextData,
      interviewType: selectedType,
      difficulty: difficultyLabel,
    };

    localStorage.setItem("interview-config", JSON.stringify(config));
    setLocation("/interview");
  };

  if (!contextData) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  const getDifficultyLabel = () => difficultyLevels.find((d) => d.value === difficultyValue[0])?.label || "Intermediate";

  return (
    <div className="container mx-auto max-w-5xl px-4 py-8">
      <ProgressStepper currentStep={2} steps={steps} />

      <div className="mt-8 space-y-6">
        <Card>
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl">Interview Type</CardTitle>
            <CardDescription>Select the type of interview you want to practice</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              {interviewTypes.map((type) => {
                const Icon = type.icon as any;
                const isSelected = selectedType === type.id;
                return (
                  <button
                    key={type.id}
                    onClick={() => setSelectedType(type.id)}
                    className={`flex flex-col items-start gap-3 rounded-lg border-2 p-4 text-left transition-all hover-elevate ${
                      isSelected ? "border-primary bg-primary/5" : "border-border bg-card"
                    }`}
                    data-testid={`button-interview-type-${type.id}`}
                  >
                    <div
                      className={`flex h-10 w-10 items-center justify-center rounded-lg ${
                        isSelected ? "bg-primary text-primary-foreground" : "bg-muted"
                      }`}
                    >
                      <Icon className="h-5 w-5" />
                    </div>
                    <div>
                      <h3 className="font-semibold">{type.title}</h3>
                      <p className="text-sm text-muted-foreground">{type.description}</p>
                    </div>
                  </button>
                );
              })}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl">Difficulty Level</CardTitle>
            <CardDescription>Adjust the complexity of interview questions</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label className="text-base font-medium">Current Level</Label>
                <Badge variant="outline" className="text-sm" data-testid="badge-difficulty">
                  {getDifficultyLabel()}
                </Badge>
              </div>
              <Slider
                value={difficultyValue}
                onValueChange={setDifficultyValue}
                max={100}
                step={50}
                className="py-4"
                data-testid="slider-difficulty"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Beginner</span>
                <span>Intermediate</span>
                <span>Advanced</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="flex items-center justify-between pt-4">
          <Button
            variant="outline"
            onClick={() => setLocation("/")}
            data-testid="button-back"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Context
          </Button>
          <Button
            size="lg"
            onClick={handleStartInterview}
            className="min-w-40"
            data-testid="button-start-interview"
          >
            Start Interview
          </Button>
        </div>
      </div>
    </div>
  );
}
