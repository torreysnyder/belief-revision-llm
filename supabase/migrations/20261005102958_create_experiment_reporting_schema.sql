create table public.experiment_runs (
    id uuid primary key,
    status text not null check (status in ('running', 'completed', 'failed')),
    started_at timestamptz not null default now(),
    completed_at timestamptz,
    target_model text not null,
    user_sim_model text not null,
    evaluator_model text not null,
    config jsonb not null default '{}'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    error text
);

create table public.conversations (
    id uuid primary key,
    run_id uuid not null references public.experiment_runs(id) on delete cascade,
    cell_id text not null,
    replicate_index integer not null check (replicate_index >= 0),
    seed bigint not null,
    vignette_id text not null,
    status text not null check (status in ('running', 'completed', 'failed')),
    started_at timestamptz not null default now(),
    completed_at timestamptz,
    style_profile jsonb not null default '{}'::jsonb,
    primary_cell jsonb not null default '{}'::jsonb,
    run_plan jsonb not null default '{}'::jsonb,
    aux_mapping jsonb not null default '{}'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    unique (run_id, cell_id, replicate_index)
);

create table public.conversation_turns (
    id bigint generated always as identity primary key,
    conversation_id uuid not null references public.conversations(id) on delete cascade,
    global_turn_index integer not null check (global_turn_index >= 0),
    speaker text not null check (speaker in ('user', 'assistant')),
    phase text not null,
    turn_in_phase integer not null check (turn_in_phase >= 0),
    text text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (conversation_id, global_turn_index, speaker)
);

create table public.behavioral_probes (
    id bigint generated always as identity primary key,
    conversation_id uuid not null references public.conversations(id) on delete cascade,
    probe_index integer not null check (probe_index >= 0),
    phase text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (conversation_id, probe_index)
);

create table public.experiment_artifacts (
    id uuid primary key,
    conversation_id uuid not null references public.conversations(id) on delete cascade,
    turn_id bigint references public.conversation_turns(id) on delete set null,
    probe_id bigint references public.behavioral_probes(id) on delete set null,
    artifact_type text not null,
    storage_bucket text not null,
    storage_path text not null,
    model_id text,
    model_revision text,
    module_path text,
    layer_index integer,
    token_start integer,
    token_end integer,
    tensor_shape integer[],
    tensor_dtype text,
    checksum text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (storage_bucket, storage_path)
);

create index conversations_run_id_status_idx
    on public.conversations (run_id, status);
create index conversation_turns_conversation_id_idx
    on public.conversation_turns (conversation_id);
create index behavioral_probes_conversation_id_idx
    on public.behavioral_probes (conversation_id);
create index experiment_artifacts_conversation_id_idx
    on public.experiment_artifacts (conversation_id);
create index experiment_artifacts_type_created_idx
    on public.experiment_artifacts (artifact_type, created_at);

alter table public.experiment_runs enable row level security;
alter table public.conversations enable row level security;
alter table public.conversation_turns enable row level security;
alter table public.behavioral_probes enable row level security;
alter table public.experiment_artifacts enable row level security;

revoke all on public.experiment_runs from anon, authenticated;
revoke all on public.conversations from anon, authenticated;
revoke all on public.conversation_turns from anon, authenticated;
revoke all on public.behavioral_probes from anon, authenticated;
revoke all on public.experiment_artifacts from anon, authenticated;

grant select, insert, update on public.experiment_runs to service_role;
grant select, insert, update on public.conversations to service_role;
grant select, insert, update on public.conversation_turns to service_role;
grant select, insert, update on public.behavioral_probes to service_role;
grant select, insert, update on public.experiment_artifacts to service_role;
grant usage, select on all sequences in schema public to service_role;
