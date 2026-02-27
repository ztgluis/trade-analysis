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

-- Drop old unique constraint on profile_name alone
ALTER TABLE custom_profiles
  DROP CONSTRAINT IF EXISTS custom_profiles_profile_name_key;

-- New unique constraint: (profile_name, workspace_id) pair must be unique
ALTER TABLE custom_profiles
  ADD CONSTRAINT custom_profiles_name_workspace_key
  UNIQUE (profile_name, workspace_id);


-- ── 2. ticker_overrides ──────────────────────────────────────────────────────

-- Add workspace_id column (existing rows get 'default')
ALTER TABLE ticker_overrides
  ADD COLUMN IF NOT EXISTS workspace_id TEXT NOT NULL DEFAULT 'default';

-- Drop old unique constraint on ticker alone
ALTER TABLE ticker_overrides
  DROP CONSTRAINT IF EXISTS ticker_overrides_ticker_key;

-- New unique constraint: (ticker, workspace_id) pair must be unique
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


-- ── 4. Verify ────────────────────────────────────────────────────────────────

-- Check that existing data is now in the 'default' workspace
SELECT 'custom_profiles' AS tbl, COUNT(*) AS rows, workspace_id FROM custom_profiles GROUP BY workspace_id
UNION ALL
SELECT 'ticker_overrides', COUNT(*), workspace_id FROM ticker_overrides GROUP BY workspace_id
UNION ALL
SELECT 'watchlist', COUNT(*), workspace_id FROM watchlist GROUP BY workspace_id;
