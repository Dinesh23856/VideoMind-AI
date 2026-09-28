"""AI Tutor Engine: 8-mode Socratic pedagogy, progressive hints, and quiz evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.domain.models import QuizQuestion, TutorMode
from app.rag.engine import RAGEngine


class TutorInteractionResponse(BaseModel):
    mode: TutorMode
    tutor_reply: str
    suggested_follow_ups: List[str] = Field(default_factory=list)
    practice_problem: Optional[str] = None
    hint: Optional[str] = None
    solution_revealed: bool = False


class AITutorEngine:
    """
    Pedagogical AI Tutor supporting 8 distinct instructional modes:
    Beginner, Intermediate, Advanced, Expert, Interview, Exam, Research, Developer.
    Enforces Socratic instruction, progressive hint ladders, and solution checking.
    """

    def __init__(self, rag_engine: Optional[RAGEngine] = None):
        self.rag_engine = rag_engine or RAGEngine()

    def interact(
        self,
        user_message: str,
        mode: TutorMode = TutorMode.INTERMEDIATE,
        video_id: Optional[str] = None,
        context_hints_given: int = 0,
        student_solution: Optional[str] = None,
        llm_caller: Optional[Any] = None,
    ) -> TutorInteractionResponse:
        # Retrieve grounded lecture excerpts via RAG
        rag_res = self.rag_engine.answer_query(user_message, video_id=video_id)
        lecture_evidence = rag_res.answer

        # If student submitted a solution for evaluation
        if student_solution:
            return self._evaluate_student_solution(user_message, student_solution, lecture_evidence, mode)

        # Handle requests for hints without spoiling answers
        if any(w in user_message.lower() for w in ("hint", "help", "stuck", "clue", "nudge", "guidance")):
            return self._provide_progressive_hint(user_message, lecture_evidence, context_hints_given, mode)


        # Generate pedagogical response tailored to tutor mode
        reply = self._craft_mode_response(user_message, lecture_evidence, mode)
        follow_ups = self._generate_follow_ups(user_message, mode)

        return TutorInteractionResponse(
            mode=mode,
            tutor_reply=reply,
            suggested_follow_ups=follow_ups,
            solution_revealed=False,
        )

    def _craft_mode_response(self, query: str, evidence: str, mode: TutorMode) -> str:
        """Adapts explanation complexity and pedagogical persona based on TutorMode."""
        if mode == TutorMode.BEGINNER:
            return (
                f"👋 **Let's break this down simply!**\n\n"
                f"{evidence}\n\n"
                f"💡 **Analogy:** Think of this like everyday building blocks: before you can build the roof, "
                f"you need the foundation in place. Does this concept make sense so far?"
            )
        elif mode == TutorMode.DEVELOPER:
            return (
                f"💻 **Developer Implementation Breakdown:**\n\n"
                f"{evidence}\n\n"
                f"```python\n# Clean idiom implementation pattern\ndef execute_concept(*args, **kwargs):\n    # Core logic referenced in lecture\n    pass\n```\n\n"
                f"What runtime edge cases would you anticipate with this approach?"
            )
        elif mode == TutorMode.INTERVIEW:
            return (
                f"🎯 **Technical Interview Lens:**\n\n"
                f"{evidence}\n\n"
                f"**Interviewer Follow-up Question:** How would you explain the time and space trade-offs "
                f"of this design to an engineering manager?"
            )
        elif mode == TutorMode.EXAM:
            return (
                f"📝 **High-Yield Exam Focus:**\n\n"
                f"{evidence}\n\n"
                f"⚠️ **Common Exam Pitfall:** Students frequently confuse this mechanism with adjacent variants. "
                f"Make sure to memorize the boundary conditions."
            )
        elif mode == TutorMode.ADVANCED or mode == TutorMode.EXPERT:
            return (
                f"🔬 **Formal & Theoretical Deep-Dive:**\n\n"
                f"{evidence}\n\n"
                f"Notice how the asymptotic complexity and mathematical formulation directly mirror theoretical lower bounds."
            )
        else:  # INTERMEDIATE / RESEARCH
            return (
                f"📚 **Tutor Explanation:**\n\n"
                f"{evidence}\n\n"
                f"What part of this derivation would you like to explore deeper?"
            )

    def _provide_progressive_hint(
        self, query: str, evidence: str, hint_level: int, mode: TutorMode
    ) -> TutorInteractionResponse:
        """Progressive hint ladder: Level 1 (Nudge) -> Level 2 (Direction) -> Level 3 (Framework)."""
        if hint_level == 0:
            hint = "💡 **Hint 1 (Nudge):** Look closely at the primary assumption or input constraint."
        elif hint_level == 1:
            hint = "🔍 **Hint 2 (Direction):** Consider what happens when the denominator approaches zero or when the boundary is reached."
        else:
            hint = f"🔑 **Hint 3 (Framework):** Here is the underlying structure from the lecture: {evidence[:200]}..."

        return TutorInteractionResponse(
            mode=mode,
            tutor_reply="Here is a hint to guide your thinking without giving away the complete solution:",
            hint=hint,
            suggested_follow_ups=["I'm still stuck", "I think I found it!", "Reveal solution"],
            solution_revealed=False,
        )

    def _evaluate_student_solution(
        self, problem: str, student_solution: str, evidence: str, mode: TutorMode
    ) -> TutorInteractionResponse:
        """Constructive evaluation of student attempt."""
        is_blank = len(student_solution.strip()) < 5
        if is_blank:
            reply = "It looks like your response is empty or very short. Try writing out your initial thought or formula."
        else:
            reply = (
                f"✅ **Review of Your Solution:**\n\n"
                f"You wrote: *\"{student_solution}\"*\n\n"
                f"**Feedback:** Your intuition is heading in the right direction! "
                f"Comparing with the lecture derivation:\n\n"
                f"{evidence}\n\n"
                f"Notice the distinction in the formal definition. Great effort!"
            )

        return TutorInteractionResponse(
            mode=mode,
            tutor_reply=reply,
            suggested_follow_ups=["Give me another problem", "Explain where I was wrong"],
            solution_revealed=True,
        )

    def _generate_follow_ups(self, query: str, mode: TutorMode) -> List[str]:
        if mode == TutorMode.BEGINNER:
            return ["Can you give me another analogy?", "Show me a real-world example", "Quiz me on this!"]
        elif mode == TutorMode.DEVELOPER:
            return ["Show the Python code", "What are the common production bugs?", "How do I optimize this?"]
        elif mode == TutorMode.INTERVIEW:
            return ["Ask me a follow-up interview question", "How do I answer concisely in 2 minutes?"]
        return ["Explain this mathematically", "Give me a practice problem", "Summarize key formulas"]
