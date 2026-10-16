drop index public.activation_captures_conversation_turn_idx;

create index activation_captures_conversation_turn_idx
    on public.activation_captures (conversation_id, global_turn_index, speaker);
