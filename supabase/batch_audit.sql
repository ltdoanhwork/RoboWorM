-- Migration for existing VGP labeling deployments.
-- Run once in Supabase SQL Editor, then ask each active rater to reload home.html.

create table if not exists public.rating_batches (
  id text primary key,
  rater_name text not null check (length(rater_name) between 1 and 60),
  version integer not null,
  reviewer boolean not null default false,
  mv_ids jsonb not null default '[]'::jsonb,
  sv_ids jsonb not null default '[]'::jsonb,
  sv_segments jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  submitted_at timestamptz,
  updated_at timestamptz not null default now()
);

alter table public.rating_batches enable row level security;

drop policy if exists "rating_batches_select_all" on public.rating_batches;
create policy "rating_batches_select_all"
  on public.rating_batches for select using (true);

drop policy if exists "rating_batches_insert_any" on public.rating_batches;
create policy "rating_batches_insert_any"
  on public.rating_batches for insert with check (true);

drop policy if exists "rating_batches_update_any" on public.rating_batches;
create policy "rating_batches_update_any"
  on public.rating_batches for update using (true);

create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists rating_batches_set_updated_at on public.rating_batches;
create trigger rating_batches_set_updated_at
  before update on public.rating_batches
  for each row execute function public.set_updated_at();
