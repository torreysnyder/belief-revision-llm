insert into storage.buckets (
    id,
    name,
    public,
    file_size_limit,
    allowed_mime_types
)
values (
    'experiment-artifacts',
    'experiment-artifacts',
    false,
    52428800,
    array['application/octet-stream']
)
on conflict (id) do update
set public = excluded.public,
    file_size_limit = excluded.file_size_limit,
    allowed_mime_types = excluded.allowed_mime_types;

create table public.activation_captures (
    id bigint generated always as identity primary key,
    artifact_id uuid not null references public.experiment_artifacts(id) on delete cascade,
    conversation_id uuid not null references public.conversations(id) on delete cascade,
    artifact_record_index integer not null check (artifact_record_index >= 0),
    activation_index integer not null check (activation_index >= 0),
    global_turn_index integer not null check (global_turn_index >= 0),
    phase text not null,
    turn_in_phase integer not null check (turn_in_phase >= 0),
    model_id text not null,
    location text not null,
    token_position text not null,
    layer_index integer not null check (layer_index >= 0),
    token_index integer not null check (token_index >= 0),
    sequence_length integer not null check (sequence_length > 0),
    tensor_shape integer[] not null,
    tensor_dtype text not null,
    speaker text generated always as ('assistant'::text) stored,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (artifact_id, artifact_record_index, activation_index),
    foreign key (conversation_id, global_turn_index, speaker)
        references public.conversation_turns (conversation_id, global_turn_index, speaker)
        on delete cascade
);

create index activation_captures_conversation_turn_idx
    on public.activation_captures (conversation_id, global_turn_index);
create index activation_captures_model_layer_idx
    on public.activation_captures (model_id, layer_index);

alter table public.activation_captures enable row level security;
revoke all on public.activation_captures from anon, authenticated;
grant select, insert, update on public.activation_captures to service_role;
grant usage, select on sequence public.activation_captures_id_seq to service_role;
