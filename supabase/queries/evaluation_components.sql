select
    r.id as run_id,
    r.started_at as run_started_at,
    r.target_model,
    r.user_sim_model,
    r.evaluator_model,
    r.config as run_config,
    c.id as conversation_id,
    c.cell_id,
    c.replicate_index,
    c.seed,
    c.vignette_id,
    c.style_profile,
    c.primary_cell,
    c.run_plan,
    c.aux_mapping,
    coalesce(turn_data.turns, '[]'::jsonb) as turns,
    coalesce(probe_data.probes, '[]'::jsonb) as probes,
    coalesce(artifact_data.artifacts, '[]'::jsonb) as artifacts
from public.conversations c
join public.experiment_runs r on r.id = c.run_id
left join lateral (
    select jsonb_agg(
        jsonb_build_object(
            'id', t.id,
            'global_turn_index', t.global_turn_index,
            'speaker', t.speaker,
            'phase', t.phase,
            'turn_in_phase', t.turn_in_phase,
            'text', t.text,
            'payload', t.payload
        ) order by t.global_turn_index, t.speaker
    ) as turns
    from public.conversation_turns t
    where t.conversation_id = c.id
) turn_data on true
left join lateral (
    select jsonb_agg(
        jsonb_build_object(
            'id', p.id,
            'probe_index', p.probe_index,
            'phase', p.phase,
            'payload', p.payload
        ) order by p.probe_index
    ) as probes
    from public.behavioral_probes p
    where p.conversation_id = c.id
) probe_data on true
left join lateral (
    select jsonb_agg(
        jsonb_build_object(
            'id', a.id,
            'artifact_type', a.artifact_type,
            'storage_bucket', a.storage_bucket,
            'storage_path', a.storage_path,
            'checksum', a.checksum,
            'metadata', a.metadata,
            'activation_captures', coalesce(capture_data.captures, '[]'::jsonb)
        ) order by a.created_at
    ) as artifacts
    from public.experiment_artifacts a
    left join lateral (
        select jsonb_agg(
            jsonb_build_object(
                'id', ac.id,
                'global_turn_index', ac.global_turn_index,
                'phase', ac.phase,
                'turn_in_phase', ac.turn_in_phase,
                'model_id', ac.model_id,
                'location', ac.location,
                'token_position', ac.token_position,
                'layer_index', ac.layer_index,
                'token_index', ac.token_index,
                'sequence_length', ac.sequence_length,
                'tensor_shape', ac.tensor_shape,
                'tensor_dtype', ac.tensor_dtype
            ) order by ac.artifact_record_index, ac.activation_index
        ) as captures
        from public.activation_captures ac
        where ac.artifact_id = a.id
    ) capture_data on true
    where a.conversation_id = c.id
) artifact_data on true
where c.status = 'completed'
order by r.started_at, c.cell_id, c.replicate_index;
