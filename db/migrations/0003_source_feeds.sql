-- 0003_source_feeds.sql — give sources an RSS feed address.
-- Run after 0002_sources.sql. Safe to re-run.
--
-- feed_url is NULLABLE on purpose: NULL means "this outlet publishes no feed and
-- needs an HTML scraper" (Imedi, Formula). That keeps the RSS half of ingestion
-- fully config-driven — adding another RSS outlet is an insert/update here, not code.

alter table sources add column if not exists feed_url text;

update sources set feed_url = 'https://netgazeti.ge/feed/' where slug = 'netgazeti';
update sources set feed_url = 'https://publika.ge/feed/'   where slug = 'publika';
update sources set feed_url = 'https://civil.ge/ka/feed'   where slug = 'civil-ge';
update sources set feed_url = 'https://on.ge/rss'          where slug = 'on-ge';

-- imedi and formula intentionally keep feed_url NULL — scrapers, added in a later task.
