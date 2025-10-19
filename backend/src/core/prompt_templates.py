"""
Prompt templates and single llm calls for interview questions.

This module provides system prompt templates for different interview types
and a template engine for dynamic prompt generation.
"""

from typing import Dict, Any, List
import regex as re
from openai import OpenAI


def create_system_prompt_templates() -> Dict[str, str]:
    """Define base system prompts for different interview types (technical, behavioral, case study, general).
    
    Returns:
        Dict[str, str]: Dictionary containing system prompt templates for each interview type.
    """
    print("[INIT] Creating system prompt templates for interview scenarios")
    
    system_prompts = {
        "technical": """You are an experienced Senior Software Engineer conducting a technical interview.
Your role is to assess the candidate's technical knowledge, problem-solving abilities, and coding skills.

Guidelines:
- Ask clear, relevant questions appropriate for the specified experience level
- Focus on practical skills and real-world applications
- Evaluate both theoretical knowledge and hands-on experience
- Provide constructive feedback when appropriate
- Maintain a professional but encouraging tone

Interview Context: Technical assessment covering programming, system design, and software engineering principles.""",

        "behavioral": """You are an experienced HR Manager conducting a behavioral interview.
Your role is to assess the candidate's soft skills, cultural fit, and professional experience.

Guidelines:
- Use the STAR method (Situation, Task, Action, Result) to structure questions
- Focus on past experiences and how they handled specific situations
- Evaluate communication skills, leadership potential, and teamwork abilities
- Ask follow-up questions to dig deeper into responses
- Maintain an empathetic and professional demeanor

Interview Context: Behavioral assessment focusing on interpersonal skills, work ethics, and cultural alignment.""",

        "case_study": """You are a Senior Business Analyst conducting a case study interview.
Your role is to evaluate the candidate's analytical thinking, business acumen, and problem-solving methodology.

Guidelines:
- Present structured business problems or scenarios
- Evaluate the candidate's approach to breaking down complex problems
- Assess their ability to think strategically and consider multiple perspectives
- Look for data-driven reasoning and logical conclusions
- Encourage questions and clarification requests

Interview Context: Case study analysis focusing on business problem-solving and strategic thinking.""",

        "general": """You are a professional interviewer conducting a comprehensive interview assessment.
Your role is to evaluate the candidate across multiple dimensions based on the job requirements.

Guidelines:
- Adapt your questioning style to the specific role and industry
- Balance technical and behavioral assessment as appropriate
- Maintain objectivity while being personable and professional
- Provide clear instructions and context for each question
- Encourage the candidate to ask clarifying questions

Interview Context: Comprehensive assessment tailored to the specific position and candidate background."""
    }
    
    print(f"[SUCCESS] Created {len(system_prompts)} system prompt templates")
    print(f"[INFO] Available templates: {list(system_prompts.keys())}")
    
    for template_type, prompt in system_prompts.items():
        if len(prompt.strip()) < 100:
            print(f"[WARNING] Template '{template_type}' may be too short ({len(prompt)} chars)")
        else:
            print(f"[VALIDATED] Template '{template_type}': {len(prompt)} characters")
    
    print("[COMPLETED] System prompt templates creation successful")
    return system_prompts


def create_user_prompt_templates() -> Dict[str, str]:
    """Design user prompt structures with placeholders for job role, experience level, question type.
    
    Returns:
        Dict[str, str]: Dictionary containing user prompt templates with placeholders for customization.
    """
    print("[INIT] Creating user prompt templates with dynamic placeholders")
    
    user_prompts = {
        "technical_question": """Generate a {difficulty_level} technical interview question for a {job_role} position.

Requirements:
- Experience Level: {experience_level}
- Focus Area: {focus_area}
- Question Type: {question_type}
- Time Limit: {time_limit} minutes

The question should be relevant to {company_context} and assess {specific_skills}.

Please provide:
1. The main question
2. Expected approach or key points to cover
3. Follow-up questions if applicable""",

        "behavioral_question": """Generate a behavioral interview question for a {job_role} candidate.

Context:
- Experience Level: {experience_level}
- Company Culture: {company_culture}
- Key Competencies: {key_competencies}
- Scenario Focus: {scenario_focus}

The question should evaluate {behavioral_traits} and be suitable for someone with {experience_level} experience.

Format the question using the STAR method framework and include guidance on what to listen for in the response.""",

        "case_study_prompt": """Create a case study scenario for a {job_role} interview.

Parameters:
- Industry: {industry}
- Business Context: {business_context}
- Problem Complexity: {complexity_level}
- Time Allocation: {time_limit} minutes
- Focus Areas: {analysis_areas}

The case should test {analytical_skills} and be appropriate for {experience_level} candidates.

Include:
1. Background information
2. The main challenge/problem
3. Available data/constraints
4. Expected deliverables""",

        "follow_up_generator": """Based on the candidate's previous response, generate {num_follow_ups} thoughtful follow-up questions.

Previous Question: {previous_question}
Candidate Response: {candidate_response}
Interview Type: {interview_type}
Areas to Explore: {exploration_areas}

Focus on:
- Clarifying ambiguous points
- Testing deeper understanding
- Exploring practical applications
- Assessing problem-solving approach

Generate questions that naturally build upon their response.""",

        "context_aware": """You are interviewing a candidate for {job_role} at {company_name}.

Job Description Summary: {job_description}
Required Skills: {required_skills}
Nice-to-Have Skills: {preferred_skills}
Team Context: {team_context}
Project Context: {project_context}

Candidate Background:
- Current Role: {current_role}
- Years of Experience: {years_experience}
- Key Technologies: {candidate_technologies}
- Previous Companies: {previous_companies}

Generate a {question_type} question that:
1. Aligns with the job requirements
2. Considers their background
3. Assesses cultural fit
4. Evaluates technical/professional growth potential"""
    }
    
    print(f"[SUCCESS] Created {len(user_prompts)} user prompt templates")
    print(f"[INFO] Available templates: {list(user_prompts.keys())}")
    
    total_placeholders = 0
    for template_name, template in user_prompts.items():
        placeholders = template.count('{')
        total_placeholders += placeholders
        print(f"[ANALYSIS] Template '{template_name}': {placeholders} placeholders, {len(template)} characters")
    
    print(f"[INFO] Total placeholders across all templates: {total_placeholders}")
    print("[COMPLETED] User prompt templates creation successful")
    
    return user_prompts


def create_context_prompt_templates() -> Dict[str, str]:
    """Build context-aware prompts that incorporate job descriptions and candidate background.
    
    Returns:
        Dict[str, str]: Dictionary containing context-aware prompt templates for enhanced interview personalization.
    """
    print("[INIT] Creating context-aware prompt templates for personalized interviews")
    
    context_prompts = {
        "job_description_integration": """Based on the following job description, create interview questions that directly assess the required qualifications:

JOB DESCRIPTION:
{job_description}

REQUIRED QUALIFICATIONS:
{required_qualifications}

PREFERRED QUALIFICATIONS:
{preferred_qualifications}

COMPANY INFORMATION:
{company_info}

Generate {num_questions} questions that:
1. Test specific technical requirements mentioned in the job description
2. Assess cultural fit based on company values
3. Evaluate experience with mentioned technologies/methodologies
4. Explore scenarios relevant to the actual role responsibilities

Interview Type: {interview_type}
Question Difficulty: {difficulty_level}""",

        "candidate_background_aware": """Considering the candidate's background, create personalized interview questions:

CANDIDATE PROFILE:
- Name: {candidate_name}
- Current Position: {current_position}
- Years of Experience: {years_experience}
- Previous Companies: {previous_companies}
- Education: {education_background}
- Key Technologies: {technical_skills}
- Notable Projects: {notable_projects}
- Career Progression: {career_progression}

TARGET ROLE:
{target_role_description}

Create questions that:
1. Bridge their current experience with the target role
2. Explore gaps and growth opportunities
3. Assess transferable skills from their background
4. Challenge them appropriately based on their experience level
5. Reference their specific experience when relevant

Focus Area: {focus_area}
Question Count: {question_count}""",

        "company_culture_context": """Incorporate company culture and values into interview questions:

COMPANY CULTURE PROFILE:
- Company Mission: {company_mission}
- Core Values: {core_values}
- Work Environment: {work_environment}
- Team Structure: {team_structure}
- Growth Philosophy: {growth_philosophy}
- Innovation Approach: {innovation_approach}

ROLE CONTEXT:
- Department: {department}
- Team Size: {team_size}
- Reporting Structure: {reporting_structure}
- Key Stakeholders: {key_stakeholders}
- Success Metrics: {success_metrics}

Generate questions that evaluate:
1. Alignment with company values and mission
2. Ability to thrive in the specific work environment
3. Collaboration style fit with team structure
4. Growth mindset compatibility
5. Innovation and problem-solving approach

Interview Focus: {culture_focus_area}""",

        "role_specific_scenarios": """Create realistic scenarios based on actual role responsibilities:

ROLE RESPONSIBILITIES:
{role_responsibilities}

TYPICAL CHALLENGES:
{typical_challenges}

STAKEHOLDER INTERACTIONS:
{stakeholder_interactions}

TECHNICAL ENVIRONMENT:
{technical_environment}

PROJECT EXAMPLES:
{project_examples}

Develop scenario-based questions that:
1. Present realistic challenges they would face in this role
2. Test decision-making in context-specific situations
3. Assess stakeholder management skills
4. Evaluate technical problem-solving in the actual tech stack
5. Explore their approach to similar challenges from their experience

Scenario Type: {scenario_type}
Complexity Level: {complexity_level}
Time Frame: {time_frame}""",

        "adaptive_depth_control": """Adjust question depth and complexity based on candidate responses and background:

INITIAL ASSESSMENT:
- Technical Level Detected: {detected_technical_level}
- Communication Style: {communication_style}
- Confidence Level: {confidence_indicators}
- Areas of Strength: {strength_areas}
- Areas for Exploration: {exploration_areas}

INTERVIEW PROGRESS:
- Questions Asked: {questions_completed}
- Strong Responses In: {strong_response_areas}
- Weak Responses In: {weak_response_areas}
- Time Remaining: {time_remaining}

Generate the next question that:
1. Appropriately challenges based on demonstrated skill level
2. Explores identified weak areas for balanced assessment
3. Builds on strong areas to showcase expertise
4. Adapts communication style to candidate preferences
5. Maintains interview flow and engagement

Next Question Focus: {next_focus_area}
Difficulty Adjustment: {difficulty_adjustment}"""
    }
    
    print(f"[SUCCESS] Created {len(context_prompts)} context-aware prompt templates")
    print(f"[INFO] Available templates: {list(context_prompts.keys())}")
    
    context_analysis = {}
    for template_name, template in context_prompts.items():
        placeholders = template.count('{')
        sections = template.count(':')
        bullet_points = template.count('1.')
        
        context_analysis[template_name] = {
            'placeholders': placeholders,
            'sections': sections,
            'structure_points': bullet_points,
            'length': len(template)
        }
        
        print(f"[ANALYSIS] Template '{template_name}': {placeholders} placeholders, {sections} sections, {len(template)} chars")
    
    total_placeholders = sum(analysis['placeholders'] for analysis in context_analysis.values())
    avg_complexity = total_placeholders / len(context_prompts)
    
    print(f"[METRICS] Total placeholders: {total_placeholders}")
    print(f"[METRICS] Average complexity per template: {avg_complexity:.1f} placeholders")
    print("[COMPLETED] Context-aware prompt templates creation successful")
    
    return context_prompts


def implement_zero_shot_prompting(
    client: OpenAI, 
    system_templates: Dict[str, str], 
    user_templates: Dict[str, str],
    interview_type: str = "technical",
    job_role: str = "Software Engineer",
    experience_level: str = "mid-level"
    ) -> Dict[str, Any]:
    """Direct questioning without examples or context.
    
    Args:
        client: OpenAI client instance
        system_templates: Dictionary of system prompt templates
        user_templates: Dictionary of user prompt templates
        interview_type: Type of interview (technical, behavioral, case_study)
        job_role: Target job role for the question
        experience_level: Candidate experience level (junior, mid-level, senior)
        
    Returns:
        Dict[str, Any]: Generated question and metadata including tokens used and response time
    """
    print(f"[INIT] Implementing zero-shot prompting for {interview_type} interview")
    print(f"[CONFIG] Role: {job_role}, Level: {experience_level}")
    
    try:
        system_prompt = system_templates.get(interview_type, system_templates["general"])
        print(f"[SETUP] Using {interview_type} system prompt ({len(system_prompt)} chars)")
        
        user_prompt = f"""Generate 1 interview question for a {job_role} position.

Requirements:
- Experience Level: {experience_level}
- Question should be clear and direct
- No examples or additional context needed
- Appropriate difficulty for the specified level

Please provide just the question without additional explanation."""
        
        print(f"[SETUP] Created zero-shot user prompt ({len(user_prompt)} chars)")
        
        print("[STEP] Executing zero-shot API call")
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=200,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        generated_question = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        print(f"[SUCCESS] Generated zero-shot question ({len(generated_question)} chars)")
        print(f"[QUESTION] {generated_question}")
        print(f"[METRICS] Tokens used: {tokens_used}")
        
        result = {
            "technique": "zero_shot",
            "interview_type": interview_type,
            "job_role": job_role,
            "experience_level": experience_level,
            "generated_question": generated_question,
            "system_prompt_used": system_prompt[:100] + "...",
            "user_prompt_used": user_prompt,
            "model_parameters": {
                "model": "gpt-4.1-nano",
                "temperature": 0.7,
                "max_tokens": 200,
                "top_p": 1.0
            },
            "metrics": {
                "total_tokens": tokens_used,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "question_length": len(generated_question)
            }
        }
        
        print("[COMPLETED] Zero-shot prompting implementation successful")
        return result
        
    except Exception as e:
        print(f"[ERROR] Zero-shot prompting failed: {str(e)}")
        raise Exception(f"Zero-shot implementation error: {str(e)}")


def generate_technical_questions(
    client: OpenAI,
    system_templates: Dict[str, str],
    job_role: str = "Software Engineer",
    experience_level: str = "mid-level",
    focus_areas: List[str] = ["coding", "system_design", "technical_knowledge"],
    num_questions: int = 3
    ) -> Dict[str, Any]:
    """Test prompts for coding, system design, and technical knowledge.
    
    Args:
        client: OpenAI client instance
        system_templates: Dictionary of system prompt templates
        job_role: Target job role for the questions
        experience_level: Candidate experience level (junior, mid-level, senior)
        focus_areas: List of technical areas to cover
        num_questions: Number of questions to generate per focus area
        
    Returns:
        Dict[str, Any]: Generated technical questions organized by focus area with metadata
    """
    print(f"[INIT] Generating technical questions for {job_role} position")
    print(f"[CONFIG] Level: {experience_level}, Focus areas: {focus_areas}, Questions per area: {num_questions}")
    
    try:
        system_prompt = system_templates.get("technical", system_templates["general"])
        print(f"[SETUP] Using technical system prompt ({len(system_prompt)} chars)")
        
        focus_specifications = {
            "coding": {
                "description": "Programming and algorithm questions",
                "topics": ["data structures", "algorithms", "code optimization", "debugging", "best practices"],
                "question_types": ["coding challenges", "code review", "algorithm explanation", "implementation problems"]
            },
            "system_design": {
                "description": "Architecture and scalability questions", 
                "topics": ["scalability", "database design", "API design", "microservices", "performance"],
                "question_types": ["architecture design", "trade-off analysis", "scaling scenarios", "system components"]
            },
            "technical_knowledge": {
                "description": "Conceptual and theoretical knowledge",
                "topics": ["frameworks", "patterns", "protocols", "security", "testing"],
                "question_types": ["concept explanation", "comparison questions", "best practice scenarios", "troubleshooting"]
            },
            "frameworks": {
                "description": "Technology-specific framework knowledge",
                "topics": ["React", "Django", "Spring", "Express", "Angular"],
                "question_types": ["implementation questions", "configuration", "optimization", "debugging"]
            },
            "devops": {
                "description": "Development operations and deployment",
                "topics": ["CI/CD", "containerization", "cloud services", "monitoring", "deployment"],
                "question_types": ["workflow design", "troubleshooting", "optimization", "tool selection"]
            }
        }
        
        generated_questions = {}
        total_tokens = 0
        
        for focus_area in focus_areas:
            if focus_area not in focus_specifications:
                print(f"[WARNING] Unknown focus area '{focus_area}', using general technical approach")
                continue
                
            spec = focus_specifications[focus_area]
            print(f"[PROCESSING] Generating {num_questions} questions for {focus_area}")
            
            user_prompt = f"""Generate {num_questions} technical interview questions for a {job_role} position focusing on {focus_area}.

Technical Focus: {spec['description']}
Key Topics: {', '.join(spec['topics'])}
Question Types: {', '.join(spec['question_types'])}
Experience Level: {experience_level}

Requirements:
- Questions should be realistic and practical
- Appropriate difficulty for {experience_level} level
- Cover different aspects of {focus_area}
- Each question should be clear and specific
- Include context where helpful

Format: Provide {num_questions} distinct questions, numbered 1-{num_questions}."""

            print(f"[STEP] Executing API call for {focus_area} questions")
            
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.2
            )
            
            questions_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            total_tokens += tokens_used
            
            generated_questions[focus_area] = {
                "specification": spec,
                "questions_text": questions_text,
                "questions_count": num_questions,
                "tokens_used": tokens_used,
                "prompt_used": user_prompt[:150] + "..."
            }
            
            print(f"[SUCCESS] Generated {focus_area} questions ({len(questions_text)} chars, {tokens_used} tokens)")
            print(f"[QUESTIONS - {focus_area.upper()}]")
            print(questions_text)
            print("-" * 60)
        
        result = {
            "technique": "technical_question_generation",
            "job_role": job_role,
            "experience_level": experience_level,
            "focus_areas_processed": focus_areas,
            "total_questions_generated": len(focus_areas) * num_questions,
            "generated_questions": generated_questions,
            "system_prompt_used": system_prompt[:100] + "...",
            "model_parameters": {
                "model": "gpt-4.1-nano",
                "temperature": 0.7,
                "max_tokens": 500,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.2
            },
            "metrics": {
                "total_tokens_used": total_tokens,
                "focus_areas_covered": len(generated_questions),
                "avg_tokens_per_area": total_tokens / len(generated_questions) if generated_questions else 0
            }
        }
        
        print("[COMPLETED] Technical question generation successful")
        print(f"[SUMMARY] Generated {len(focus_areas) * num_questions} questions across {len(focus_areas)} technical areas")
        print(f"[METRICS] Total tokens used: {total_tokens}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Technical question generation failed: {str(e)}")
        raise Exception(f"Technical question generation error: {str(e)}")

def generate_behavioral_questions(
    client: OpenAI,
    system_templates: Dict[str, str],
    user_templates: Dict[str, str],
    job_role: str = "Software Engineer",
    experience_level: str = "mid-level",
    competencies: List[str] = None,
    num_questions: int = 3
    ) -> Dict[str, Any]:
    """Test prompts for soft skills and experience-based questions.
    
    Args:
        client: OpenAI client instance
        system_templates: Dictionary of system prompt templates
        user_templates: Dictionary of user prompt templates
        job_role: Target job role for the questions
        experience_level: Candidate experience level
        competencies: List of specific competencies to assess
        num_questions: Number of behavioral questions to generate
        
    Returns:
        Dict[str, Any]: Generated behavioral questions and assessment metadata
    """
    print(f"[INIT] Generating behavioral questions for {job_role} role")
    print(f"[CONFIG] Level: {experience_level}, Questions: {num_questions}")
    
    if competencies is None:
        competencies = ["teamwork", "leadership", "problem_solving", "communication", "adaptability"]
    
    print(f"[COMPETENCIES] Assessing: {', '.join(competencies)}")
    
    try:
        system_prompt = system_templates.get("behavioral", system_templates["general"])
        print(f"[SETUP] Using behavioral system prompt ({len(system_prompt)} chars)")
        
        user_prompt = f"""Generate {num_questions} behavioral interview questions for a {job_role} position.

        Requirements:
        - Experience Level: {experience_level}
        - Focus on these competencies: {', '.join(competencies)}
        - Use STAR method framework (Situation, Task, Action, Result)
        - Questions should elicit specific past experiences
        - Appropriate for {experience_level} level expectations

        For each question, structure it to:
        1. Ask for a specific situation or experience
        2. Encourage detailed storytelling
        3. Allow assessment of the target competency
        4. Be realistic and commonly asked in actual interviews

        Generate questions that an HR manager or hiring manager would actually ask."""
        
        print(f"[SETUP] Created behavioral prompt targeting {len(competencies)} competencies ({len(user_prompt)} chars)")
        
        print("[STEP] Executing behavioral questions generation")
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,
            max_tokens=500,
            top_p=0.9,
            frequency_penalty=0.3,
            presence_penalty=0.2
        )
        
        generated_questions = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        questions_list = []
        lines = generated_questions.split('\n')
        current_question = ""
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                if current_question:
                    questions_list.append(current_question.strip())
                current_question = line
            elif line and current_question:
                current_question += " " + line
        
        if current_question:
            questions_list.append(current_question.strip())
        
        print(f"[SUCCESS] Generated {len(questions_list)} behavioral questions ({len(generated_questions)} chars)")
        print(f"[QUESTIONS] {generated_questions}")
        print(f"[COMPETENCIES] Questions designed to assess: {', '.join(competencies)}")
        print("[FRAMEWORK] STAR method structure for detailed responses")
        print(f"[METRICS] Tokens used: {tokens_used}")
        
        result = {
            "technique": "behavioral_questions",
            "job_role": job_role,
            "experience_level": experience_level,
            "target_competencies": competencies,
            "num_questions_requested": num_questions,
            "num_questions_generated": len(questions_list),
            "generated_questions": generated_questions,
            "questions_list": questions_list,
            "system_prompt_used": system_prompt[:100] + "...",
            "user_prompt_used": user_prompt[:200] + "...",
            "model_parameters": {
                "model": "gpt-4.1-nano",
                "temperature": 0.6,
                "max_tokens": 500,
                "top_p": 0.9,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.2
            },
            "metrics": {
                "total_tokens": tokens_used,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "questions_length": len(generated_questions),
                "avg_question_length": len(generated_questions) / max(len(questions_list), 1)
            }
        }
        
        print("[COMPLETED] Behavioral questions generation successful")
        return result
        
    except Exception as e:
        print(f"[ERROR] Behavioral questions generation failed: {str(e)}")
        raise Exception(f"Behavioral questions error: {str(e)}")


def implement_response_scorer(
    client: OpenAI,
    question: str,
    interview_type: str = "technical",
    job_role: str = "Software Engineer",
    experience_level: str = "mid-level",
    scoring_criteria: List[str] = None
    ) -> Dict[str, Any]:
    """Create scoring function for relevance, clarity, and difficulty.
    
    Args:
        client: OpenAI client instance
        question: Interview question to score
        interview_type: Type of interview (technical, behavioral, case_study)
        job_role: Target job role for context
        experience_level: Candidate experience level
        scoring_criteria: Custom scoring criteria (defaults to standard set)
        
    Returns:
        Dict[str, Any]: Comprehensive scoring results with metrics and feedback
    """
    print(f"[INIT] Implementing response scorer for {interview_type} question")
    print(f"[CONFIG] Role: {job_role}, Level: {experience_level}")
    
    if scoring_criteria is None:
        scoring_criteria = ["relevance", "clarity", "difficulty", "specificity", "actionability"]
    
    try:
        scoring_prompt = f"""Evaluate the following interview question across multiple quality dimensions:

    QUESTION TO EVALUATE:
    "{question}"

    CONTEXT:
    - Interview Type: {interview_type}
    - Job Role: {job_role}
    - Experience Level: {experience_level}

    Please score this question on a scale of 1-10 for each criterion and provide brief justification:

    1. RELEVANCE (1-10): How relevant is this question to the {job_role} role and {interview_type} interview context?

    2. CLARITY (1-10): How clear and well-structured is the question? Is it easy to understand what's being asked?

    3. DIFFICULTY (1-10): Is the difficulty level appropriate for {experience_level} candidates?

    4. SPECIFICITY (1-10): How specific and focused is the question? Does it target particular skills/knowledge?

    5. ACTIONABILITY (1-10): Does the question allow for demonstrable, assessable responses?

    Format your response as:
    RELEVANCE: [score]/10 - [brief justification]
    CLARITY: [score]/10 - [brief justification]  
    DIFFICULTY: [score]/10 - [brief justification]
    SPECIFICITY: [score]/10 - [brief justification]
    ACTIONABILITY: [score]/10 - [brief justification]

    OVERALL_SCORE: [average]/10
    STRENGTHS: [key strengths]
    IMPROVEMENTS: [suggested improvements]"""

        print(f"[SETUP] Created comprehensive scoring prompt ({len(scoring_prompt)} chars)")
        
        print("[STEP] Executing AI-assisted question scoring")
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert interview designer and talent assessment specialist. Provide objective, detailed evaluations of interview questions based on industry best practices."
                },
                {"role": "user", "content": scoring_prompt}
            ],
            temperature=0.3,
            max_tokens=500,
            top_p=0.9
        )
        
        scoring_response = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        print(f"[SUCCESS] Generated comprehensive scoring analysis ({len(scoring_response)} chars)")
        
        scores = {}
        score_pattern = r'(\w+):\s*(\d+)/10\s*-\s*(.+?)(?=\n|$)'
        matches = re.findall(score_pattern, scoring_response, re.IGNORECASE | re.MULTILINE)
        
        for criterion, score, justification in matches:
            scores[criterion.lower()] = {
                'score': int(score),
                'justification': justification.strip()
            }
        
        overall_match = re.search(r'OVERALL_SCORE:\s*(\d+(?:\.\d+)?)/10', scoring_response)
        overall_score = float(overall_match.group(1)) if overall_match else 0
        
        strengths_match = re.search(r'STRENGTHS:\s*(.+?)(?=\nIMPROVEMENTS:|$)', scoring_response, re.DOTALL)
        strengths = strengths_match.group(1).strip() if strengths_match else "Not specified"
        
        improvements_match = re.search(r'IMPROVEMENTS:\s*(.+?)$', scoring_response, re.DOTALL)
        improvements = improvements_match.group(1).strip() if improvements_match else "Not specified"
        
        question_length = len(question)
        word_count = len(question.split())
        has_specific_requirements = any(keyword in question.lower() for keyword in ['explain', 'describe', 'implement', 'design', 'analyze'])
        
        print(f"[ANALYSIS] Parsed {len(scores)} scoring criteria")
        print(f"[METRICS] Overall Score: {overall_score}/10")
        print(f"[METRICS] Question Length: {question_length} chars, {word_count} words")
        print(f"[ASSESSMENT] Specific Requirements Detected: {has_specific_requirements}")
        
        result = {
            "scoring_technique": "ai_assisted_evaluation",
            "question_evaluated": question,
            "context": {
                "interview_type": interview_type,
                "job_role": job_role,
                "experience_level": experience_level
            },
            "scores": scores,
            "overall_score": overall_score,
            "strengths": strengths,
            "improvements": improvements,
            "question_metrics": {
                "length_chars": question_length,
                "word_count": word_count,
                "has_specific_requirements": has_specific_requirements,
                "criteria_evaluated": len(scores)
            },
            "ai_response": {
                "full_analysis": scoring_response,
                "tokens_used": tokens_used,
                "model": "gpt-4.1-nano"
            }
        }
        
        print("[SCORES] Quality Assessment:")
        for criterion, data in scores.items():
            print(f"  {criterion.upper()}: {data['score']}/10 - {data['justification'][:50]}...")
        
        print("[COMPLETED] Response scoring implementation successful")
        return result
        
    except Exception as e:
        print(f"[ERROR] Response scoring failed: {str(e)}")
        raise Exception(f"Scoring implementation error: {str(e)}")


def assess_role_alignment(
    client: OpenAI,
    generated_question: str,
    job_requirements: Dict[str, Any],
    job_role: str,
    experience_level: str,
    assessment_criteria: List[str] = None
    ) -> Dict[str, Any]:
    """Verify questions match specified job requirements.
    
    Args:
        client: OpenAI client instance
        generated_question: The interview question to assess
        job_requirements: Dictionary containing job requirements, skills, responsibilities
        job_role: Target job role
        experience_level: Required experience level
        assessment_criteria: Optional list of specific criteria to evaluate
        
    Returns:
        Dict[str, Any]: Alignment assessment with scores and detailed analysis
    """
    print(f"[INIT] Assessing role alignment for {job_role} question")
    print(f"[CONFIG] Experience level: {experience_level}")
    
    try:
        if assessment_criteria is None:
            assessment_criteria = [
                "technical_skill_relevance",
                "experience_level_appropriateness", 
                "job_responsibility_alignment",
                "industry_context_accuracy",
                "practical_applicability"
            ]
        
        print(f"[SETUP] Using {len(assessment_criteria)} assessment criteria")
        
        required_skills = job_requirements.get("required_skills", [])
        preferred_skills = job_requirements.get("preferred_skills", [])
        responsibilities = job_requirements.get("responsibilities", [])
        industry_context = job_requirements.get("industry", "technology")
        
        print(f"[ANALYSIS] Job requirements: {len(required_skills)} required skills, {len(responsibilities)} responsibilities")
        
        assessment_prompt = f"""Analyze how well this interview question aligns with the specified job requirements.

    INTERVIEW QUESTION TO ASSESS:
    "{generated_question}"

    JOB ROLE: {job_role}
    EXPERIENCE LEVEL: {experience_level}

    JOB REQUIREMENTS:
    Required Skills: {', '.join(required_skills)}
    Preferred Skills: {', '.join(preferred_skills)}
    Key Responsibilities: {', '.join(responsibilities)}
    Industry Context: {industry_context}

    IMPORTANT: Provide scores in exactly this format for each criterion:

    1. Technical Skill Relevance: X/10 - [justification]
    2. Experience Level Appropriateness: X/10 - [justification]  
    3. Job Responsibility Alignment: X/10 - [justification]
    4. Industry Context Accuracy: X/10 - [justification]
    5. Practical Applicability: X/10 - [justification]

    Replace X with your score (1-10). Be specific about how the question relates to the job requirements."""
            
        print(f"[SETUP] Created alignment assessment prompt ({len(assessment_prompt)} chars)")
        
        print("[STEP] Executing role alignment assessment")
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are an expert HR analyst. Provide scores in the exact format requested: 'Criterion Name: X/10 - justification'"},
                {"role": "user", "content": assessment_prompt}
            ],
            temperature=0.3,
            max_tokens=600,
            top_p=0.9
        )
        
        assessment_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        print(f"[SUCCESS] Generated alignment assessment ({len(assessment_text)} chars)")
        
        scores = {}
        total_score = 0
        
        pattern1 = r'(\w+(?:\s+\w+)*?):\s*(\d+)\/10'
        matches1 = re.findall(pattern1, assessment_text, re.IGNORECASE)
        
        pattern2 = r'(\d+)\/10.*?([A-Z][a-zA-Z\s]+?):'
        matches2 = re.findall(pattern2, assessment_text, re.IGNORECASE)
        
        pattern3 = r'(?:Technical|Experience|Job|Industry|Practical).*?(\d+)\/10'
        matches3 = re.findall(pattern3, assessment_text, re.IGNORECASE)
        
        print(f"[DEBUG] Pattern 1 matches: {len(matches1)}")
        print(f"[DEBUG] Pattern 2 matches: {len(matches2)}")
        print(f"[DEBUG] Pattern 3 matches: {len(matches3)}")
        
        if matches1:
            for criterion, score in matches1:
                criterion_clean = criterion.lower().replace(' ', '_')
                score_int = int(score)
                scores[criterion_clean] = score_int
                total_score += score_int
        elif matches3:
            criteria_names = ["technical_skill_relevance", "experience_level_appropriateness", 
                            "job_responsibility_alignment", "industry_context_accuracy", "practical_applicability"]
            for i, score in enumerate(matches3[:5]):
                if i < len(criteria_names):
                    scores[criteria_names[i]] = int(score)
                    total_score += int(score)
        
        max_possible_score = len(scores) * 10 if scores else 50
        alignment_percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        
        if alignment_percentage >= 80:
            alignment_level = "HIGH"
        elif alignment_percentage >= 60:
            alignment_level = "MODERATE"
        elif alignment_percentage >= 40:
            alignment_level = "LOW"
        else:
            alignment_level = "POOR"
        
        print(f"[ASSESSMENT] Overall alignment: {alignment_level} ({alignment_percentage:.1f}%)")
        print(f"[SCORES] Individual criterion scores: {scores}")
        print(f"[DEBUG] Assessment text preview: {assessment_text[:200]}...")
        print(f"[METRICS] Tokens used: {tokens_used}")
        
        result = {
            "technique": "role_alignment_assessment",
            "question_assessed": generated_question,
            "job_role": job_role,
            "experience_level": experience_level,
            "job_requirements": job_requirements,
            "assessment_criteria": assessment_criteria,
            "detailed_assessment": assessment_text,
            "criterion_scores": scores,
            "total_score": total_score,
            "max_possible_score": max_possible_score,
            "alignment_percentage": round(alignment_percentage, 1),
            "alignment_level": alignment_level,
            "model_parameters": {
                "model": "gpt-4.1-nano",
                "temperature": 0.3,
                "max_tokens": 600
            },
            "metrics": {
                "total_tokens": tokens_used,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "assessment_length": len(assessment_text)
            }
        }
        
        print("[COMPLETED] Role alignment assessment successful")
        return result
        
    except Exception as e:
        print(f"[ERROR] Role alignment assessment failed: {str(e)}")
        raise Exception(f"Role alignment assessment error: {str(e)}")

def implement_chain_of_thought(
    client: OpenAI,
    system_templates: Dict[str, str],
    user_templates: Dict[str, str],
    interview_type: str = "technical",
    job_role: str = "Software Engineer",
    experience_level: str = "mid-level",
    complexity_level: str = "moderate"
    ) -> Dict[str, Any]:
    """Step-by-step reasoning prompts for complex scenarios.
    
    Args:
        client: OpenAI client instance
        system_templates: Dictionary of system prompt templates
        user_templates: Dictionary of user prompt templates
        interview_type: Type of interview (technical, behavioral, case_study)
        job_role: Target job role for the question
        experience_level: Candidate experience level (junior, mid-level, senior)
        complexity_level: Scenario complexity (simple, moderate, complex)
        
    Returns:
        Dict[str, Any]: Generated question with step-by-step reasoning structure
    """
    print(f"[INIT] Implementing chain-of-thought prompting for {interview_type} interview")
    print(f"[CONFIG] Role: {job_role}, Level: {experience_level}, Complexity: {complexity_level}")
    
    try:
        system_prompt = system_templates.get(interview_type, system_templates["general"])
        print(f"[SETUP] Using {interview_type} system prompt ({len(system_prompt)} chars)")
        
        user_prompt = f"""Generate 1 interview question for a {job_role} position that requires step-by-step reasoning.

        Requirements:
        - Experience Level: {experience_level}
        - Complexity: {complexity_level}
        - The question should encourage the candidate to think through the problem step-by-step
        - Include clear reasoning checkpoints or sub-questions
        - Guide the candidate through a logical problem-solving process

        Structure the question to include:
        1. A clear problem statement or scenario
        2. Step-by-step guidance or sub-questions that build upon each other
        3. Reasoning checkpoints that help break down the complex problem
        4. Encouragement for the candidate to "think out loud" and explain their reasoning

        Example of chain-of-thought structure:
        "Let's think through this step by step:
        1. First, consider...
        2. Next, think about...
        3. Then, evaluate...
        4. Finally, conclude..."

        Generate a question that naturally guides this type of structured thinking process."""
        
        print(f"[SETUP] Created chain-of-thought user prompt ({len(user_prompt)} chars)")
        
        print("[STEP] Executing chain-of-thought API call with step-by-step reasoning structure")
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=400,
            top_p=0.8,
            frequency_penalty=0.1,
            presence_penalty=0.2
        )
        
        generated_question = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        step_indicators = ["step", "first", "next", "then", "finally", "consider", "think", "evaluate"]
        reasoning_elements = sum(1 for indicator in step_indicators if indicator.lower() in generated_question.lower())
        
        print(f"[SUCCESS] Generated chain-of-thought question with structured reasoning ({len(generated_question)} chars)")
        print(f"[QUESTION] {generated_question}")
        print(f"[REASONING] Question includes {reasoning_elements} step-by-step reasoning elements")
        print("[STRUCTURE] Designed to guide candidate through logical problem-solving process")
        print(f"[METRICS] Tokens used: {tokens_used}")
        
        result = {
            "technique": "chain_of_thought",
            "interview_type": interview_type,
            "job_role": job_role,
            "experience_level": experience_level,
            "complexity_level": complexity_level,
            "generated_question": generated_question,
            "reasoning_elements_detected": reasoning_elements,
            "system_prompt_used": system_prompt[:100] + "...",
            "user_prompt_used": user_prompt[:200] + "...",
            "model_parameters": {
                "model": "gpt-4.1-nano",
                "temperature": 0.5,
                "max_tokens": 400,
                "top_p": 0.8,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2
            },
            "metrics": {
                "total_tokens": tokens_used,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "question_length": len(generated_question),
                "reasoning_complexity": reasoning_elements
            }
        }
        
        print("[COMPLETED] Chain-of-thought prompting implementation successful")
        return result
        
    except Exception as e:
        print(f"[ERROR] Chain-of-thought prompting failed: {str(e)}")
        raise Exception(f"Chain-of-thought implementation error: {str(e)}")


def implement_few_shot_prompting(
    client: OpenAI,
    system_templates: Dict[str, str],
    user_templates: Dict[str, str],
    interview_type: str = "technical",
    job_role: str = "Software Engineer",
    experience_level: str = "mid-level",
    num_examples: int = 3
    ) -> Dict[str, Any]:
    """Include 2-3 example Q&A pairs for pattern learning.
    
    Args:
        client: OpenAI client instance
        system_templates: Dictionary of system prompt templates
        user_templates: Dictionary of user prompt templates
        interview_type: Type of interview (technical, behavioral, case_study)
        job_role: Target job role for the question
        experience_level: Candidate experience level (junior, mid-level, senior)
        num_examples: Number of example Q&A pairs to include (2-3)
        
    Returns:
        Dict[str, Any]: Generated question and metadata including examples used
    """
    print(f"[INIT] Implementing few-shot prompting for {interview_type} interview")
    print(f"[CONFIG] Role: {job_role}, Level: {experience_level}, Examples: {num_examples}")
    
    try:
        system_prompt = system_templates.get(interview_type, system_templates["general"])
        print(f"[SETUP] Using {interview_type} system prompt ({len(system_prompt)} chars)")
        
        examples = {
            "technical": [
                {
                    "question": "Explain the difference between `==` and `===` in JavaScript and when you would use each.",
                    "good_answer": "== performs type coercion before comparison, while === checks both value and type without coercion. Use === for strict equality to avoid unexpected results."
                },
                {
                    "question": "How would you optimize a slow database query that returns customer data?",
                    "good_answer": "I'd analyze the execution plan, add appropriate indexes, consider query restructuring, and potentially implement caching or pagination for large result sets."
                },
                {
                    "question": "Describe the differences between REST and GraphQL APIs.",
                    "good_answer": "REST uses multiple endpoints and HTTP methods, while GraphQL uses a single endpoint with flexible queries. GraphQL allows clients to request specific fields, reducing over-fetching."
                }
            ],
            "behavioral": [
                {
                    "question": "Tell me about a time when you had to work with a difficult team member.",
                    "good_answer": "I focused on understanding their perspective, communicated openly about project goals, found common ground, and established clear expectations that improved our collaboration."
                },
                {
                    "question": "Describe a situation where you had to learn a new technology quickly.",
                    "good_answer": "When our team adopted React, I created a learning plan with documentation, tutorials, and hands-on projects. I also paired with experienced developers and built a small prototype."
                },
                {
                    "question": "How do you handle competing priorities and tight deadlines?",
                    "good_answer": "I assess urgency and impact, communicate with stakeholders about trade-offs, break tasks into smaller chunks, and regularly update on progress while maintaining quality standards."
                }
            ],
            "case_study": [
                {
                    "question": "Our e-commerce site has a 40% cart abandonment rate. How would you investigate and address this?",
                    "good_answer": "I'd analyze user behavior data, identify drop-off points, test checkout flow, examine payment options, and implement A/B tests for improvements like guest checkout or progress indicators."
                },
                {
                    "question": "Design a system to handle 1 million concurrent users for a social media platform.",
                    "good_answer": "I'd implement load balancing, database sharding, caching layers, CDN for static content, microservices architecture, and horizontal scaling with auto-scaling groups."
                },
                {
                    "question": "A key team member just left and took critical knowledge with them. How do you handle this?",
                    "good_answer": "Immediate: document their work, identify knowledge gaps, redistribute responsibilities. Long-term: implement knowledge sharing practices, documentation standards, and cross-training programs."
                }
            ]
        }
        
        selected_examples = examples.get(interview_type, examples["technical"])[:num_examples]
        print(f"[EXAMPLES] Selected {len(selected_examples)} example Q&A pairs for pattern learning")
        
        examples_text = "\n\n".join([
            f"Example {i+1}:\nQ: {ex['question']}\nGood Answer: {ex['good_answer']}"
            for i, ex in enumerate(selected_examples)
        ])
        
        user_prompt = f"""Based on the following examples, generate 1 similar interview question for a {job_role} position.

        {examples_text}

        Now generate a new question that:
        - Follows the same style and complexity level as the examples
        - Is appropriate for {experience_level} experience level
        - Tests similar competencies but with different content
        - Maintains the quality and depth shown in the examples

        Please provide just the question without additional explanation."""
        
        print(f"[SETUP] Created few-shot user prompt with {num_examples} examples ({len(user_prompt)} chars)")
        
        print("[STEP] Executing few-shot API call with example patterns")
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,
            max_tokens=250,
            top_p=0.9,
            frequency_penalty=0.2,
            presence_penalty=0.1
        )
        
        generated_question = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        print(f"[SUCCESS] Generated few-shot question using {num_examples} example patterns ({len(generated_question)} chars)")
        print(f"[QUESTION] {generated_question}")
        print("[PATTERN] Examples provided pattern learning for consistent style and complexity")
        print(f"[METRICS] Tokens used: {tokens_used}")
        
        result = {
            "technique": "few_shot",
            "interview_type": interview_type,
            "job_role": job_role,
            "experience_level": experience_level,
            "num_examples": num_examples,
            "examples_used": selected_examples,
            "generated_question": generated_question,
            "system_prompt_used": system_prompt[:100] + "...",
            "user_prompt_used": user_prompt[:200] + "...",
            "model_parameters": {
                "model": "gpt-4.1-nano",
                "temperature": 0.6,
                "max_tokens": 250,
                "top_p": 0.9,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.1
            },
            "metrics": {
                "total_tokens": tokens_used,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "question_length": len(generated_question),
                "examples_provided": len(selected_examples)
            }
        }
        
        print("[COMPLETED] Few-shot prompting implementation successful")
        return result
        
    except Exception as e:
        print(f"[ERROR] Few-shot prompting failed: {str(e)}")
        raise Exception(f"Few-shot implementation error: {str(e)}")

def implement_role_based_prompting(
    client: OpenAI,
    system_templates: Dict[str, str],
    user_templates: Dict[str, str],
    interviewer_persona: str = "senior_engineer",
    job_role: str = "Software Engineer",
    experience_level: str = "mid-level",
    company_context: str = "tech_startup"
    ) -> Dict[str, Any]:
    """Assign specific interviewer personas (senior engineer, HR manager, etc.)
    
    Args:
        client: OpenAI client instance
        system_templates: Dictionary of system prompt templates
        user_templates: Dictionary of user prompt templates
        interviewer_persona: Specific interviewer role/persona
        job_role: Target job role for the question
        experience_level: Candidate experience level
        company_context: Company/industry context
        
    Returns:
        Dict[str, Any]: Generated question with specific interviewer perspective
    """
    print(f"[INIT] Implementing role-based prompting with {interviewer_persona} persona")
    print(f"[CONFIG] Role: {job_role}, Level: {experience_level}, Context: {company_context}")
    
    try:
        personas = {
            "senior_engineer": {
                "background": "Senior Software Engineer with 8+ years experience",
                "focus": "technical depth, coding practices, system design",
                "style": "direct, practical, focuses on real-world scenarios",
                "priorities": "code quality, scalability, problem-solving approach"
            },
            "tech_lead": {
                "background": "Technical Lead managing a team of 5-8 developers",
                "focus": "technical leadership, architecture decisions, team collaboration",
                "style": "strategic thinking, balances technical and people skills",
                "priorities": "system architecture, mentoring ability, technical decision-making"
            },
            "hr_manager": {
                "background": "HR Manager specializing in technical recruiting",
                "focus": "cultural fit, communication skills, career motivation",
                "style": "empathetic, behavioral-focused, people-oriented",
                "priorities": "team dynamics, growth mindset, company values alignment"
            },
            "product_manager": {
                "background": "Product Manager with technical background",
                "focus": "business impact, user-focused thinking, cross-functional collaboration",
                "style": "business-minded, user-centric, metrics-driven",
                "priorities": "product thinking, stakeholder management, business value"
            },
            "startup_founder": {
                "background": "Startup Founder/CTO building early-stage company",
                "focus": "versatility, ownership mindset, rapid execution",
                "style": "entrepreneurial, fast-paced, resource-conscious",
                "priorities": "adaptability, initiative, building from scratch"
            },
            "enterprise_architect": {
                "background": "Enterprise Architect at large corporation",
                "focus": "enterprise patterns, compliance, large-scale systems",
                "style": "methodical, governance-focused, risk-aware",
                "priorities": "enterprise standards, security, maintainability"
            }
        }
        
        persona_details = personas.get(interviewer_persona, personas["senior_engineer"])
        print(f"[PERSONA] Selected: {persona_details['background']}")
        print(f"[FOCUS] Interview focus: {persona_details['focus']}")
        
        role_based_system = f"""You are a {persona_details['background']} conducting an interview.

        Your interviewing style: {persona_details['style']}
        Your key priorities: {persona_details['priorities']}
        Your focus areas: {persona_details['focus']}

        Company context: {company_context}

        Conduct the interview from this specific perspective, asking questions that reflect your role, experience, and priorities. Your questions should feel authentic to someone in your position and reflect the concerns and interests typical of your role."""
                
        user_prompt = f"""Generate 1 interview question for a {job_role} candidate from your perspective as a {persona_details['background']}.

        Requirements:
        - Experience Level: {experience_level}
        - Reflect your specific role's concerns and priorities
        - Use language and focus areas typical of your position
        - Consider what YOU would actually want to know about this candidate
        - Make it feel authentic to your interviewing style: {persona_details['style']}

        The question should assess: {persona_details['priorities']}

        Generate a realistic question that you would actually ask in this interview context."""
        
        print(f"[SETUP] Created role-based prompt for {interviewer_persona} perspective ({len(user_prompt)} chars)")
        
        print(f"[STEP] Executing role-based API call from {interviewer_persona} perspective")
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": role_based_system},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=300,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.3
        )
        
        generated_question = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        print(f"[SUCCESS] Generated role-based question from {interviewer_persona} perspective ({len(generated_question)} chars)")
        print(f"[QUESTION] {generated_question}")
        print(f"[PERSPECTIVE] Question reflects {persona_details['background']} priorities and style")
        print(f"[AUTHENTICITY] Designed to feel realistic for {interviewer_persona} interviewer")
        print(f"[METRICS] Tokens used: {tokens_used}")
        
        result = {
            "technique": "role_based",
            "interviewer_persona": interviewer_persona,
            "persona_details": persona_details,
            "job_role": job_role,
            "experience_level": experience_level,
            "company_context": company_context,
            "generated_question": generated_question,
            "system_prompt_used": role_based_system[:200] + "...",
            "user_prompt_used": user_prompt[:200] + "...",
            "model_parameters": {
                "model": "gpt-4.1-nano",
                "temperature": 0.7,
                "max_tokens": 300,
                "top_p": 0.9,
                "presence_penalty": 0.3
            },
            "metrics": {
                "total_tokens": tokens_used,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "question_length": len(generated_question)
            }
        }
        
        print("[COMPLETED] Role-based prompting implementation successful")
        return result
        
    except Exception as e:
        print(f"[ERROR] Role-based prompting failed: {str(e)}")
        raise Exception(f"Role-based implementation error: {str(e)}")


class PromptTemplateEngine:
    """A simple template engine for interview prompts."""
    
    def __init__(self, templates: Dict[str, str] = None):
        """Initialize the template engine.
        
        Args:
            templates (Dict[str, str], optional): Template dictionary. Defaults to None.
        """
        self.templates = templates or create_system_prompt_templates()
    
    def get_template(self, template_type: str) -> str:
        """Get a template by type.
        
        Args:
            template_type (str): Type of template to retrieve
            
        Returns:
            str: The template string
        """
        return self.templates.get(template_type, self.templates.get("general", ""))
    
    def list_templates(self) -> list:
        """List all available template types.
        
        Returns:
            list: List of available template types
        """
        return list(self.templates.keys())
    
    def render_template(self, template_type: str, **kwargs) -> str:
        """Render a template with provided context variables.
        
        Args:
            template_type (str): Type of template to render
            **kwargs: Context variables to substitute in the template
            
        Returns:
            str: Rendered template with variables substituted
        """
        template = self.get_template(template_type)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required template variable: {e}")
        except Exception as e:
            raise ValueError(f"Error rendering template: {str(e)}")
    
    def add_template(self, template_type: str, template_content: str) -> None:
        """Add a new template to the engine.
        
        Args:
            template_type (str): Type identifier for the template
            template_content (str): Template content string
        """
        self.templates[template_type] = template_content
    
    def remove_template(self, template_type: str) -> bool:
        """Remove a template from the engine.
        
        Args:
            template_type (str): Type identifier for the template to remove
            
        Returns:
            bool: True if template was removed, False if it didn't exist
        """
        if template_type in self.templates:
            del self.templates[template_type]
            return True
        return False
    
    def update_template(self, template_type: str, template_content: str) -> None:
        """Update an existing template or add it if it doesn't exist.
        
        Args:
            template_type (str): Type identifier for the template
            template_content (str): New template content string
        """
        self.templates[template_type] = template_content
    
    def template_exists(self, template_type: str) -> bool:
        """Check if a template exists in the engine.
        
        Args:
            template_type (str): Type identifier to check
            
        Returns:
            bool: True if template exists, False otherwise
        """
        return template_type in self.templates
