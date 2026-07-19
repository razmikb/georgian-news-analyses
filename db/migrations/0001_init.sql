-- 0001_init.sql — Ground News Georgia initial schema
-- Run once in the Supabase SQL Editor (paste whole file → Run).
-- Creates: pgvector extension, leaning enum, sources/events/articles tables,
-- indexes, and row-level security (public can read, only the pipeline can write).

-- Embeddings: Gemini `text-embedding-004` outputs 768-dimensional vectors.
-- If we ever switch embedding models, this dimension (and the column below) changes.
create extension if not exists vector;

-- The four editorial leaning buckets. See PLAN.md §2/§4 — these labels are the
-- product's core taxonomy. Adding a value later: `alter type source_leaning add value '...';`
do $$
begin
  if not exists (select 1 from pg_type where typname = 'source_leaning') then
    create type source_leaning as enum ('pro_gov', 'opposition', 'independent', 'center');
  end if;
end$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- sources — the news outlets. Leaning labels are manual editorial data (config),
-- revisable in the DB; the public methodology page explains who assigned them.
-- ─────────────────────────────────────────────────────────────────────────────
create table if not exists sources (
  id        bigint generated always as identity primary key,
  slug      text          not null unique,
  name_ka   text          not null,
  name_en   text          not null,
  url       text          not null,
  leaning   source_leaning not null,
  logo_url  text
);

-- ─────────────────────────────────────────────────────────────────────────────
-- events — a real-world story that one or more articles cover. Titles generated
-- in both Georgian and English. leaning_coverage / is_blindspot filled in Phase 3.
-- ─────────────────────────────────────────────────────────────────────────────
create table if not exists events (
  id               bigint generated always as identity primary key,
  title_ka         text,
  title_en         text,
  first_seen_at    timestamptz not null default now(),
  last_updated_at  timestamptz not null default now(),
  article_count    integer     not null default 0,
  leaning_coverage jsonb,
  is_blindspot     boolean     not null default false
);

-- ─────────────────────────────────────────────────────────────────────────────
-- articles — one headline + link per row. We store headlines and links ONLY,
-- never full article text (copyright). `url` is unique — that is how we dedupe.
-- embedding is nullable: filled after fetch, and pruned to NULL past ~7 days
-- (only the 48h window needs it for clustering).
-- ─────────────────────────────────────────────────────────────────────────────
create table if not exists articles (
  id           bigint generated always as identity primary key,
  source_id    bigint      not null references sources (id) on delete cascade,
  event_id     bigint      references events (id) on delete set null,
  headline     text        not null,
  url          text        not null unique,
  published_at timestamptz,
  embedding    vector(768),
  fetched_at   timestamptz not null default now()
);

create index if not exists articles_event_id_idx     on articles (event_id);
create index if not exists articles_published_at_idx  on articles (published_at desc);
create index if not exists events_last_updated_at_idx  on events (last_updated_at desc);

-- Approximate-nearest-neighbour index for the 48h cosine similarity search.
-- HNSW needs no training data (works from row zero), unlike ivfflat.
create index if not exists articles_embedding_idx
  on articles using hnsw (embedding vector_cosine_ops);

-- ─────────────────────────────────────────────────────────────────────────────
-- Row-level security: the frontend queries the DB directly with the public
-- `anon` key. RLS makes that safe — anon may SELECT everything and write nothing.
-- The pipeline writes with the secret `service_role` key, which bypasses RLS.
-- ─────────────────────────────────────────────────────────────────────────────
alter table sources  enable row level security;
alter table events   enable row level security;
alter table articles enable row level security;

drop policy if exists "public read" on sources;
drop policy if exists "public read" on events;
drop policy if exists "public read" on articles;

create policy "public read" on sources  for select using (true);
create policy "public read" on events   for select using (true);
create policy "public read" on articles for select using (true);
