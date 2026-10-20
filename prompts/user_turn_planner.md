You are planning the NEXT user turn in a natural conversation.

Hidden user state:
{{user_state_json}}

Fixed style profile:
{{style_profile_json}}

Recent conversation:
{{recent_dialogue_json}}

Core claim:
{{core_claim}}

Displayed auxiliaries:
A1: {{displayed_A1}}
A2: {{displayed_A2}}

Current situation:
- phase: {{phase_name}}
- turn in phase: {{turn_number}} of {{phase_turn_count}}
- selected dialogue act: {{dialogue_act}}
- conversation goal: {{conversation_goal}}
- belief anchor level: {{belief_anchor_level}}
- evidence role surfaced now: {{evidence_role}}
- evidence text: {{evidence_text}}
- active auxiliaries already mentioned: {{active_auxiliaries_json}}

Task:
Produce a compact JSON plan for the next user turn.

Rules:
- keep the user's current anchor in the core claim as the default baseline
- if core-disconfirming evidence appears, react realistically: resist, wobble, question, or reinterpret
- if auxiliary-related evidence appears, you may shift toward or away from an auxiliary explanation
- only lean on an auxiliary if it is psychologically motivated by the conversation
- do not sound like a polished debater
- keep the next turn brief and natural

Return JSON only with this schema:
{
  "dialogue_act": "string",
  "emotion": "string",
  "belief_move": "hold|slight_wobble|slight_strengthen|qualified_retreat",
  "should_mention_evidence": true,
  "should_ask_question": false,
  "focus": "what the user is mainly reacting to",
  "content_notes": "1-2 short notes for what to mention"
}
