with base as (
    {{ query:evaluation_components }}
)
select
    base.*,
    jsonb_array_length(base.turns) as turn_count,
    jsonb_array_length(base.probes) as probe_count,
    coalesce((
        select sum(jsonb_array_length(artifact -> 'activation_captures'))
        from jsonb_array_elements(base.artifacts) as artifact
        where artifact -> 'metadata' ->> 'extension' = 'torrey-caa'
    ), 0) as activation_capture_count
from base
where exists (
    select 1
    from jsonb_array_elements(base.artifacts) as artifact
    where artifact -> 'metadata' ->> 'extension' = 'torrey-caa'
)
order by base.run_started_at desc, base.cell_id, base.replicate_index;
