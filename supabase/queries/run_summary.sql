select
    run.id as run_id,
    run.status,
    run.started_at,
    run.completed_at,
    run.target_model,
    run.user_sim_model,
    run.evaluator_model,
    count(conversation.id) as conversation_count
from public.experiment_runs as run
left join public.conversations as conversation
    on conversation.run_id = run.id
group by run.id
order by run.started_at desc;
