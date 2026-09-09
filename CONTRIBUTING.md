# Contributing

This document defines how changes flow into this repository. Both human contributors and Claude Code should follow it.

## Branch model

```
feature/* ──► dev ──► main (tagged release)
```

- **`main`** — release branch. Tagged with semver. Never receive direct commits.
- **`dev`** — integration branch. Direct commits allowed for small fixes; feature branches merge here first.
- **`feature/*`** — short-lived, per-topic branches. Branch off `dev`, rebase onto `dev` during development, merge back to `dev` via a merge-commit PR, then delete. One topic → one branch → one PR.

### Never rewrite shared history

`main` and `dev` — and any **integration feature branch** (one with sub-branches based on it, see Workflow 4) — are *shared*. Never rebase, force-push, or amend already-pushed commits on them. Rebase rewrites commit SHAs, so anyone who already pulled the old commits ends up with a divergent local history that is painful to reconcile.

Branches you alone work on (typical short-lived `feature/*` branches) *may* be rebased — that's why Workflow 2 says "rebase onto `dev`, then `--force-with-lease`." The rule of thumb: if anyone else might have based work on a branch, treat it as shared and don't rewrite it.

## Four workflows

### 1. Small change → commit directly to `dev`

For typos, config tweaks, single-file fixes, log-level changes, etc.

```text
    main (tag: 0.1.5)
        v
    o---o
         \
          B---C    <- dev
          ^   ^
          |   [tweak] log level
          [fix] typo
```

```bash
git checkout dev
git pull --ff-only origin dev
# ... edit, stage ...
git commit -m "[type] short description"
git push origin dev
```

### 2. Feature work → feature branch → PR to `dev`

For anything non-trivial: new ETL, new dashboard, refactor, multi-file change.

```text
                     W---W---P    <- feature/new-metric (delete after merge)
                    /
    o---o------o---S              <- dev
        ^      ^   ^
        |      |   [feat] new metric (PR merge commit)
        |      [fix] log level
        feature branched here
```

While the feature branch is in progress, `dev` may receive new commits (like `[fix] log level` above). Before pushing or opening a PR, **rebase the feature branch onto current `dev`** so it stays based on the latest tip:

```text
Before  `git rebase origin/dev`:

    A---B---C        <- dev (advanced while you were working)
         \
          D---E      <- feature/new-metric (still based on B)

After:

    A---B---C            <- dev
             \
              D'---E'    <- feature/new-metric (rewritten on top of C)
```

The original `D` and `E` are discarded; `D'` and `E'` are new commits with the same diffs but new SHAs. This is why the next push needs `--force-with-lease` — the remote feature branch's old commits are no longer in the history.

```bash
# start
git checkout dev && git pull --ff-only origin dev
git checkout -b feature/my-thing

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

**Merge to `dev`: merge commit** (squash-merge is disabled on this repo). The PR's individual commits land on `dev` under a merge commit, so keep the branch's commits clean (one logical concern each — see the commit guidelines).

- GitHub UI: pick "Merge pull request" on the PR; check the "Delete branch" option.
- Terminal: `gh pr merge <num> --merge --delete-branch`

**Always delete the feature branch after the merge**, locally too (`git branch -d <branch>`; `git fetch --prune` cleans up remote-tracking refs). Every merged branch is fully recoverable from its merge commit and PR, so kept branches are pure clutter. Start the next piece of work from a fresh branch off current `dev`.

```bash
# delete locally if not already done by gh
git checkout dev && git pull --ff-only origin dev
git branch -D feature/my-thing

# next piece of work: branch off fresh dev
git checkout -b feature/next-thing
```

If your branch naming scheme has been per-developer (e.g. `nx/dashboard-work`) rather than per-topic, switch to per-topic names (`feat/dashboard-num-bets`, `fix/etl-time-range`). Per-topic names map cleanly to one PR → one merge → delete.

### 3. Release: `dev` → `main` with tag

Done after a meaningful batch of work has accumulated on `dev`.

```text
    main (tag: 0.1.5)                  main (tag: 0.1.6)
        v                                  v
    o---o-----------------M    <- after Step B & C: M is the --no-ff merge commit
         \               /
          F1---F2---C            <- dev (F1, F2 = features; C = [docs] changelog 0.1.6)

    After Step D (fast-forward dev to match main):

    o---o-----------------M    <- main, dev (both at 0.1.6)
         \               /
          F1---F2---C
```

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

### 4. Multi-person feature → sub-branches → integration branch → PR to `dev`

For a feature too large for one person, branch a long-lived **integration feature branch** off `dev`, then have each developer work on a short-lived **sub-branch** that fans into it. Sub-branches follow Workflow 2 (rebase + merge-commit PR); the integration branch is *shared* — no history rewrites.

```text
    dev          o---o---o---o
                  \           \
    feature/big    I---I'---I''---M    <- integration (merge-commit PR into dev)
                    \   ^   ^
                     \  |   |
    feature/big/A     a-a'  |          (merge-commit PR into feature/big)
                            |
    feature/big/B           b---b'     (merge-commit PR into feature/big)
```

```bash
# 1. create the integration branch off dev (one-time, by whoever starts)
git checkout dev && git pull --ff-only origin dev
git checkout -b feature/big
git push -u origin feature/big

# 2. each contributor: sub-branch off the integration branch
git checkout feature/big && git pull --ff-only origin feature/big
git checkout -b feature/big/component-a

# 3. develop on the sub-branch; rebase onto feature/big (NOT dev)
git fetch origin
git rebase origin/feature/big
git push --force-with-lease origin feature/big/component-a

# 4. open a PR from sub-branch into the integration branch, merge, delete
gh pr create --base feature/big --title "[feat] component A of big-thing"
gh pr merge <num> --merge --delete-branch

# 5. keep the integration branch current with dev — MERGE, don't rebase
#    (rebase would rewrite SHAs that other sub-branches are based on)
git checkout feature/big && git pull --ff-only origin feature/big
git fetch origin
git merge origin/dev
git push origin feature/big

# 6. when the whole feature is ready: PR feature/big → dev, merge as usual
gh pr create --base dev --title "[feat] big-thing"
gh pr merge <num> --merge --delete-branch
```

**When to choose Workflow 4 over Workflow 2**

- **Default to Workflow 2.** Single owner, single short branch — simplest path, smallest review surface.
- **Reach for Workflow 4** only when the feature is genuinely too big for one person *and* decomposes into independently-reviewable pieces. The cost is a long-lived integration branch that drifts from `dev` and needs periodic `git merge origin/dev` to stay current; the longer it lives, the more painful the eventual merge back to `dev`.
- **Avoid two people pushing directly to one shared feature branch.** It works for a day or two, then becomes a worse version of Workflow 4 (every push is a coordination event, conflicts pile up, no isolated review). Switch to sub-branches as soon as the feature outgrows one person.

## Pull request descriptions

Every PR (feature → dev, sub-branch → integration branch, or dev → main) gets a structured body. Claude Code will draft this when asked.

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
