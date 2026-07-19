-- 0002_sources.sql — seed the 6 MVP sources (PLAN.md §4).
-- Run after 0001_init.sql. Safe to re-run: `on conflict (slug) do nothing`.
--
-- name_ka spellings are best-effort — a native Georgian speaker should verify them.
-- logo_url left NULL for now (added in Phase 4).

insert into sources (slug, name_ka, name_en, url, leaning) values
  ('netgazeti', 'ნეტგაზეთი', 'Netgazeti',  'https://netgazeti.ge',   'independent'),
  ('publika',   'პუბლიკა',   'Publika',    'https://publika.ge',     'independent'),
  ('civil-ge',  'სივილ ჯორჯია', 'Civil.ge', 'https://civil.ge',      'independent'),
  ('on-ge',     'ონ.ჯი',     'On.ge',      'https://on.ge',          'independent'),
  ('imedi',     'იმედი',     'Imedi News', 'https://imedinews.ge',   'pro_gov'),
  ('formula',   'ფორმულა',   'Formula',    'https://formulanews.ge', 'opposition')
on conflict (slug) do nothing;
