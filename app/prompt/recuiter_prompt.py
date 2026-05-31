from langchain_core.prompts import ChatPromptTemplate

# ── System Prompt Template ─────────────────

RECRUITER_SYSTEM_PROMPT = """
You are an expert AI Recruiter assistant with deep knowledge of:
- Resume and CV analysis
- Job description matching
- Candidate evaluation
- Skills assessment
- Experience verification

Your responsibilities:
- Answer questions based ONLY on the provided context
- Extract relevant information from CVs and job descriptions
- Match candidates to job requirements accurately
- Provide professional and unbiased assessments
- If information is not in the context, say 'I cannot find relevant information'

Always be:
- Professional and objective
- Concise and accurate
- Unbiased in candidate evaluation
"""


JOB_MATCH_SYSTEM_PROMPT = """
You are an expert AI Recruiter specializing in job matching.
Your task is to match candidates to job descriptions based on:
- Required skills match
- Experience level match
- Education requirements
- Job type preference

Scoring guidelines:
- 0.9 - 1.0 : Excellent match
- 0.7 - 0.9 : Good match
- 0.5 - 0.7 : Partial match
- 0.0 - 0.5 : Poor match

Always provide:
- Match score (0.0 to 1.0)
- Matched skills list
- Brief summary of match quality
- Answer based ONLY on provided context
"""


CANDIDATE_EXTRACTION_PROMPT = """
You are an expert CV analyzer.
Extract the following information from the CV context:
- Full name
- Email address
- Phone number
- Skills (as a list)
- Years of experience
- Experience level (entry/junior/mid/senior/lead)
- Education
- Job type preference

Return ONLY the extracted information.
If a field is not found, return null for that field.

"""

# ── Recuiter Prompt Template ────────────────
def get_recruiter_prompt() -> ChatPromptTemplate:
    """ Return the system prompt template for the recruiter assistant
    """
    return ChatPromptTemplate.from_messages([
        ('system', RECRUITER_SYSTEM_PROMPT),
        ('human', "Context:\n{context} \n\n Question: {question}"),
    ])


# ── Job Match Prompt Template ───────────────
def get_job_match_prompt() -> ChatPromptTemplate:
    """ Return the system prompt template for job matching
    """
    return ChatPromptTemplate.from_messages([
        ('system', JOB_MATCH_SYSTEM_PROMPT),
        ('human', "Job Description Context:{job_context} \n\n "
         "Candidate CV Context:{candidate_context} \n\n "
         "Evaluate the match between the candidate and the job description and provide a match score, matched skills and a brief summary."
         ),
    ])

# ── Candidate Extraction Prompt Template ───────────────
def get_candidate_extraction_prompt() -> ChatPromptTemplate:
    """ Return the system prompt template for candidate information extraction
    """
    return ChatPromptTemplate.from_messages([
        ('system', CANDIDATE_EXTRACTION_PROMPT),
        ('human', "CV Context:{candidate_context} \n\n Extract the required information from the candidate's CV based on the system prompt instructions."),
    ])

# ── Prompt Registry ─────────────────
PROMPT_REGISTRY = {
    "recruiter": get_recruiter_prompt,
    "job_match": get_job_match_prompt,
    "candidate_extraction": get_candidate_extraction_prompt
}

def get_prompt(prompt_name: str = "recruiter") -> ChatPromptTemplate:
    """ Get the prompt template by name from the registry
    """
    prompt  = PROMPT_REGISTRY.get(prompt_name)
    if not prompt:
        raise ValueError(
            f"Prompt '{prompt_name}' not found in registry"
            f"Available prompts: {list(PROMPT_REGISTRY.keys())}")

    return PROMPT_REGISTRY[prompt_name]()


__all__ = [
    "RECRUITER_SYSTEM_PROMPT",
    "JOB_MATCH_SYSTEM_PROMPT",
    "CANDIDATE_EXTRACTION_PROMPT",
    "PROMPT_REGISTRY",
    "get_recruiter_prompt",
    "get_job_match_prompt",
    "get_candidate_extraction_prompt",
    "get_prompt",
]
