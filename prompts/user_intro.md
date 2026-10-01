You are planning the user's FIRST message.

Hidden user state:
{{user_state_json}}

Recent conversation:
{{recent_dialogue_json}}

Core claim:
{{core_claim}}

Task:
Generate the user's first message.
The first message must contain:
1. a clear statement of the user's current belief about the core claim,
2. a brief personal stake or lived context,
3. a move inviting the assistant to respond.

Constraints:
- sound natural and conversational
- usually 1-2 sentences, max 3
- do not sound polished or essay-like
- stay consistent with style profile: {{style_profile_json}}

Output only the next message.
