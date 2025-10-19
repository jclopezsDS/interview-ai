import { useState } from "react";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Sparkles } from "lucide-react";
import { ProgressStepper } from "@/components/progress-stepper";

const steps = [
  { id: 1, title: "Context", description: "Provide details" },
  { id: 2, title: "Configure", description: "Set preferences" },
  { id: 3, title: "Interview", description: "Practice live" },
];

export default function ContextCollection() {
  const [, setLocation] = useLocation();
  const [jobDescription, setJobDescription] = useState("");
  const [candidateBackground, setCandidateBackground] = useState("");
  

  const canProceed = jobDescription.trim().length > 20 && candidateBackground.trim().length > 20;

  const handleContinue = () => {
    const contextData = {
      jobDescription,
      candidateBackground,
    };
    localStorage.setItem("interview-context", JSON.stringify(contextData));
    setLocation("/configure");
  };

  return (
    <div className="container mx-auto max-w-5xl px-4 py-8">
      <ProgressStepper currentStep={1} steps={steps} />
      
      <Card className="mt-8">
        <CardHeader className="space-y-1 pb-6">
          <CardTitle className="flex items-center gap-2 text-2xl md:text-3xl">
            <Sparkles className="h-6 w-6 text-primary" />
            Tell Us About Your Interview
          </CardTitle>
          <CardDescription className="text-base">
            Provide context about the role and your background so we can create a personalized interview experience
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <div className="space-y-3">
              <Label htmlFor="job-description" className="text-sm font-medium">
                Job Description <span className="text-destructive">*</span>
              </Label>
              <Textarea
                id="job-description"
                placeholder="Paste the job description here. Include role requirements, responsibilities, and required skills..."
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                className="min-h-60 resize-none"
                data-testid="input-job-description"
              />
              <p className="text-xs text-muted-foreground">
                {jobDescription.length} characters
              </p>
            </div>

            <div className="space-y-3">
              <Label htmlFor="candidate-background" className="text-sm font-medium">
                Your Background <span className="text-destructive">*</span>
              </Label>
              <Textarea
                id="candidate-background"
                placeholder="Share your professional background, relevant experience, key skills, and achievements..."
                value={candidateBackground}
                onChange={(e) => setCandidateBackground(e.target.value)}
                className="min-h-60 resize-none"
                data-testid="input-candidate-background"
              />
              <p className="text-xs text-muted-foreground">
                {candidateBackground.length} characters
              </p>
            </div>
          </div>

          {/* Optional fields removed for MVP */}

          <div className="flex justify-end pt-4">
            <Button
              size="lg"
              onClick={handleContinue}
              disabled={!canProceed}
              className="min-w-40"
              data-testid="button-continue"
            >
              Continue to Configuration
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
