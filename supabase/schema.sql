-- Video World labeling schema — Supabase (Postgres), 100% free tier.
--
-- Design choice: NO Supabase Auth (its built-in email provider is capped at
-- ~2 emails/hour on the free plan, which stalls if several raters sign up at
-- once). Instead: a fixed account allowlist and passcode gate live client-side.
-- RLS is fully open (any anon-key holder can read/write `labels`) — acceptable
-- for a small trusted internal study, NOT a real security boundary. Do not put
-- anything sensitive in this table.
--
-- Run this once in the Supabase SQL editor (Project -> SQL Editor -> New query).

create table if not exists public.labels (
  rater_name text not null check (length(rater_name) between 1 and 60),
  item_id text not null,
  score smallint not null check (score between 1 and 3),
  updated_at timestamptz not null default now()
);

alter table public.labels
  add constraint labels_pkey primary key (rater_name, item_id);

alter table public.labels enable row level security;

-- Open read: needed so the (passcode-gated) frontend can show cross-rater
-- progress and the admin export page without a backend.
create policy "labels_select_all"
  on public.labels for select
  using (true);

create policy "labels_insert_any"
  on public.labels for insert
  with check (true);

create policy "labels_update_any"
  on public.labels for update
  using (true);

-- Keep updated_at fresh on every upsert.
create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists labels_set_updated_at on public.labels;
create trigger labels_set_updated_at
  before update on public.labels
  for each row execute function public.set_updated_at();

-- Exact server-side batch manifest. This preserves each rater's displayed
-- Geometry/Hallucinate order for audit and correction by batch index.
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

drop trigger if exists rating_batches_set_updated_at on public.rating_batches;
create trigger rating_batches_set_updated_at
  before update on public.rating_batches
  for each row execute function public.set_updated_at();
