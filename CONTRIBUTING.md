# Contributing

This document defines how changes flow into this repository. Both human contributors and Claude Code should follow it.

## Branch model

```
feature/* ──► dev ──► main (tagged release)
```

- **`main`** — release branch. Tagged with semver. Never receive direct commits.
- **`dev`** — integration branch. Direct commits allowed for small fixes; feature branches merge here first.
- **`feature/*`** — topic branches. Branch off `dev`, rebase onto `dev` during development, merge back to `dev` via PR.

## Three workflows

### 1. Small change → commit directly to `dev`

For typos, config tweaks, single-file fixes, log-level changes, etc.

```bash
git checkout dev
git pull --ff-only origin dev
# ... edit, stage ...
git commit -m "[type] short description"
git push origin dev
```

### 2. Feature work → feature branch → PR to `dev`

For anything non-trivial: new ETL, new dashboard, refactor, multi-file change.

```bash
# start
git checkout dev && git pull --ff-only origin dev
git checkout -b feature/my-thing       # or reuse an existing personal branch

# develop, committing as you go
git add ... && git commit -m "[feat] ..."

# stay current with dev (rebase, do NOT merge dev into feature)
git fetch origin
git rebase origin/dev
# resolve conflicts if any, then:
git push --force-with-lease origin feature/my-thing

# open PR (terminal) — base = dev
gh pr create --base dev --title "[feat] ..." --body "..."
```

**Merge to `dev`: squash-merge.** Each feature becomes one clean commit on `dev`.

- GitHub UI: pick "Squash and merge" on the PR.
- Terminal: `gh pr merge <num> --squash --delete-branch=false`

Feature branches are **not deleted**. After merge, before starting new work on the same branch, rebase it onto fresh `dev`:
```bash
git checkout feature/my-thing
git fetch origin
git reset --hard origin/dev    # if no in-flight work; otherwise rebase
```

### 3. Release: `dev` → `main` with tag

Done after a meaningful batch of work has accumulated on `dev`.

**Step A — update `CHANGELOG.md` on `dev`**

Pick the next version (semver):
- **patch** (0.1.5 → 0.1.6): bug fixes, small tweaks
- **minor** (0.1.6 → 0.2.0): new features, non-breaking changes
- **major** (0.2.0 → 1.0.0): breaking changes

Move items from `## [Unreleased]` into a new `## [X.Y.Z] - YYYY-MM-DD` section. Commit on `dev`:

```bash
git checkout dev
git pull --ff-only origin dev
# ... edit CHANGELOG.md ...
git commit -m "[docs] changelog for X.Y.Z"
git push origin dev
```

To see what landed since the last tag:
```bash
git log 0.1.5..dev --oneline
```

**Step B — merge `dev` → `main` (no-ff merge commit)**

Terminal:
```bash
git checkout main
git pull --ff-only origin main
git merge --no-ff dev -m "Release X.Y.Z"
git push origin main
```

GitHub UI: open a PR from `dev` to `main`, pick **"Create a merge commit"** (NOT squash). Merge.

**Step C — tag the merge commit**

Annotated tag with the same content as the changelog section:

Terminal:
```bash
git checkout main && git pull --ff-only origin main
git tag -a X.Y.Z -m "$(sed -n '/^## \[X.Y.Z\]/,/^## \[/p' CHANGELOG.md | sed '$d')"
git push origin X.Y.Z
```

GitHub UI: Releases → "Draft a new release" → choose tag `X.Y.Z`, target `main`, paste the changelog section into the description, publish.

**Step D — fast-forward `dev` to match `main`** (so they don't drift):

```bash
git checkout dev
git merge --ff-only main
git push origin dev
```

## Pull request descriptions

Every PR (feature → dev, or dev → main) gets a structured body. Claude Code will draft this when asked.

```markdown
## Summary
1–3 sentences on the *why*.

## Changes
- file/area: what changed and why
- file/area: what changed and why

## Test plan
- [ ] manual check 1
- [ ] manual check 2
- [ ] tests run / linter clean

## Breaking changes
None / describe migration path.

## Related
Issue/ticket links if any.
```

## Commit message format

Defined in `CLAUDE.md` → **Commit Guidelines**. Summary:

```
[type] short imperative description
```

Types: `feat` · `fix` · `chore` · `refactor` · `docs` · `test` · `tweak`

One commit per logical concern. Don't split a single file across commits.

## Changelog format

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each released version has a section grouped into:

- **Added** — new features
- **Changed** — changes to existing behavior
- **Fixed** — bug fixes
- **Removed** — removed features
- **Deprecated** — soon-to-be-removed features
- **Security** — security fixes

`[Unreleased]` at the top accumulates entries between releases.

## Tag scheme

Semantic versioning, no `v` prefix (matches existing tags `0.1.0`–`0.1.5`):

```
MAJOR.MINOR.PATCH
```

Tags are **annotated** (use `git tag -a`, not lightweight) and carry the changelog section as the message.

## Asking Claude Code to merge

You can ask Claude to handle any of the three flows. Examples:

- *"Commit this change directly to dev"* → flow 1
- *"Open a PR from this feature branch to dev"* → flow 2
- *"Release 0.1.6: merge dev into main and tag"* → flow 3 (Claude will draft changelog from `git log`, ask you to confirm version bump, then execute)

Claude will pause for confirmation before any push to `main`, any tag push, or any force-push.
