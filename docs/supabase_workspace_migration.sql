-- ─────────────────────────────────────────────────────────────────────────────
-- Workspace Migration: Add multi-user support via workspace_id
--
-- Run this in Supabase SQL Editor (Settings → SQL Editor → New Query)
-- Safe to run once. Existing data is migrated to the 'default' workspace.
-- ─────────────────────────────────────────────────────────────────────────────


-- ── 1. custom_profiles ───────────────────────────────────────────────────────

-- Add workspace_id column (existing rows get 'default')
ALTER TABLE custom_profiles
  ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default';


-- ── 2. ticker_overrides ──────────────────────────────────────────────────────

-- Add workspace_id column (existing rows get 'default')
ALTER TABLE ticker_overrides
  ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default';

-- Drop the FK first (it depends on custom_profiles_profile_name_key index)
ALTER TABLE ticker_overrides
  DROP CONSTRAINT IF EXISTS ticker_overrides_profile_name_fkey;

-- Now safe to drop the old unique constraint on profile_name alone
ALTER TABLE custom_profiles
  DROP CONSTRAINT IF EXISTS custom_profiles_profile_name_key;

-- New unique constraint on custom_profiles: (profile_name, workspace_id)
ALTER TABLE custom_profiles
  ADD CONSTRAINT custom_profiles_name_workspace_key
  UNIQUE (profile_name, workspace_id);

-- New FK on ticker_overrides referencing the compound key
ALTER TABLE ticker_overrides
  ADD CONSTRAINT ticker_overrides_profile_workspace_fkey
  FOREIGN KEY (profile_name, workspace_id)
  REFERENCES custom_profiles (profile_name, workspace_id)
  ON DELETE CASCADE;

-- New unique constraint on ticker_overrides: (ticker, workspace_id)
ALTER TABLE ticker_overrides
  DROP CONSTRAINT IF EXISTS ticker_overrides_ticker_key;

ALTER TABLE ticker_overrides
  ADD CONSTRAINT ticker_overrides_ticker_workspace_key
  UNIQUE (ticker, workspace_id);


-- ── 3. watchlist ─────────────────────────────────────────────────────────────

-- Add workspace_id column (existing rows get 'default')
ALTER TABLE watchlist
  ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default';

-- Drop old primary key (was id=1 singleton)
-- Add unique constraint on workspace_id so each workspace has one watchlist row
ALTER TABLE watchlist
  ADD CONSTRAINT watchlist_workspace_key
  UNIQUE (workspace_id);


-- ── 4. saved_strategies (NEW — safe to run on fresh or existing installs) ────

CREATE TABLE IF NOT EXISTS saved_strategies (
  id           BIGSERIAL  PRIMARY KEY,
  workspace_id TEXT       NOT NULL DEFAULT 'default',
  name         TEXT       NOT NULL,
  code         TEXT       NOT NULL,
  profile_name TEXT,
  indicators   JSONB      DEFAULT '[]',
  entry_mode   TEXT       DEFAULT 'all_signals',
  timeframe    TEXT       DEFAULT 'D',
  created_at   TIMESTAMP  DEFAULT NOW(),
  updated_at   TIMESTAMP  DEFAULT NOW(),
  CONSTRAINT saved_strategies_name_workspace_key UNIQUE (name, workspace_id)
);


-- ── 5. Verify ────────────────────────────────────────────────────────────────

-- Check that existing data is now in the 'default' workspace
SELECT 'custom_profiles' AS tbl, COUNT(*) AS rows, workspace_id FROM custom_profiles GROUP BY workspace_id
UNION ALL
SELECT 'ticker_overrides', COUNT(*), workspace_id FROM ticker_overrides GROUP BY workspace_id
UNION ALL
SELECT 'watchlist', COUNT(*), workspace_id FROM watchlist GROUP BY workspace_id
UNION ALL
SELECT 'saved_strategies', COUNT(*), workspace_id FROM saved_strategies GROUP BY workspace_id;
