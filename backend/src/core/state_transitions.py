"""
State transition management for interview conversations.

This module handles the state machine logic for transitioning between
different phases of an interview conversation.

Note:
- The app runtime uses the pure functions defined here (allowed_transitions,
  validate_transition, initial_phase, next_phase_on_event).
- A legacy wrapper `implement_state_transitions()` is preserved to keep
  notebooks report-ready. It is not used by the FastAPI app.
"""

from enum import Enum
from typing import Dict, List
from datetime import datetime


class InterviewPhase(Enum):
    GREETING = "greeting"
    QUESTIONING = "questioning"
    EVALUATION = "evaluation"
    CONCLUSION = "conclusion"

def allowed_transitions() -> Dict[InterviewPhase, List[InterviewPhase]]:
    """Return the allowed transitions graph for interview phases."""
    return {
        InterviewPhase.GREETING: [InterviewPhase.QUESTIONING],
        InterviewPhase.QUESTIONING: [InterviewPhase.EVALUATION, InterviewPhase.CONCLUSION],
        InterviewPhase.EVALUATION: [InterviewPhase.QUESTIONING, InterviewPhase.CONCLUSION],
        InterviewPhase.CONCLUSION: [],
    }


def validate_transition(current_phase: str, target_phase: str) -> bool:
    """Validate if transition is allowed (string-safe API for routers/services)."""
    try:
        current = InterviewPhase(current_phase)
        target = InterviewPhase(target_phase)
    except ValueError:
        return False
    return target in allowed_transitions()[current]


def initial_phase() -> InterviewPhase:
    """Initial phase for a new session."""
    return InterviewPhase.GREETING


def next_phase_on_event(
    phase: InterviewPhase,
    event: str,
    turns: int,
    max_turns: int = 3,
    ) -> InterviewPhase:
    """Deterministic phase transition policy for MVP.

    - On session start (event 'start'): GREETING -> QUESTIONING
    - During QUESTIONING/EVALUATION, alternate until `turns` >= `max_turns`, then -> CONCLUSION
    - On event 'end' from any non-conclusion state -> CONCLUSION
    """
    if event == "end":
        return InterviewPhase.CONCLUSION

    if phase == InterviewPhase.GREETING:
        # On start or first answer, move to questioning
        return (
            InterviewPhase.QUESTIONING
            if event in {"start", "answer"}
            else InterviewPhase.GREETING
        )

    if phase in {InterviewPhase.QUESTIONING, InterviewPhase.EVALUATION}:
        if turns >= max_turns:
            return InterviewPhase.CONCLUSION
        # Alternate between questioning and evaluation on each user turn
        return (
            InterviewPhase.EVALUATION
            if phase == InterviewPhase.QUESTIONING
            else InterviewPhase.QUESTIONING
        )

    return InterviewPhase.CONCLUSION


# ===== Legacy, notebook-friendly API (not used by the app) =====
def implement_state_transitions() -> Dict[str, object]:
    """Legacy wrapper returned structure for notebooks.

    Returns a dict of helpers similar to the original experimental API,
    backed by the new pure functions in this module.
    """
    vt = allowed_transitions()

    def _validate_transition(current_phase: str, target_phase: str) -> bool:
        return validate_transition(current_phase, target_phase)

    def _get_next_phases(current_phase: str) -> List[str]:
        try:
            cur = InterviewPhase(current_phase)
        except ValueError:
            return []
        return [p.value for p in vt[cur]]

    def _is_terminal_state(phase: str) -> bool:
        try:
            cur = InterviewPhase(phase)
        except ValueError:
            return False
        return len(vt[cur]) == 0

    def _transition_state(state: object, target_phase: str, reason: str = "") -> bool:
        """Best-effort state mutation for notebooks (duck-typed).

        Expects `state` to have attributes or dict-keys: current_phase, updated_at, context_data.
        """
        try:
            current_phase = getattr(state, "current_phase", None) or state.get("current_phase")  # type: ignore[attr-defined]
            if not _validate_transition(str(current_phase), target_phase):
                return False

            # mutate phase
            if hasattr(state, "current_phase"):
                setattr(state, "current_phase", target_phase)
            else:
                state["current_phase"] = target_phase  # type: ignore[index]

            # updated_at
            ts = datetime.utcnow().isoformat()
            if hasattr(state, "updated_at"):
                setattr(state, "updated_at", ts)
            else:
                state["updated_at"] = ts  # type: ignore[index]

            # transitions log (optional)
            try:
                ctx = getattr(state, "context_data", None) or state.get("context_data")  # type: ignore[attr-defined]
                if ctx is not None:
                    ctx.setdefault("transitions", []).append({  # type: ignore[call-arg]
                        "from": current_phase,
                        "to": target_phase,
                        "timestamp": ts,
                        "reason": reason,
                    })
            except Exception:
                pass

            return True
        except Exception:
            return False

    return {
        "valid_transitions": {k.value: [v.value for v in vt[k]] for k in vt},
        "validate_transition": _validate_transition,
        "transition_state": _transition_state,
        "get_next_phases": _get_next_phases,
        "is_terminal": _is_terminal_state,
        "phases": [p.value for p in InterviewPhase],
    }