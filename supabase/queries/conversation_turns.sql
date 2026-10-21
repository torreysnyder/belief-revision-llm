select turn.*
from public.conversation_turns as turn
where turn.conversation_id = %(conversation_id)s::uuid
order by turn.global_turn_index, turn.speaker;
