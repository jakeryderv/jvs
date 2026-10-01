#!/usr/bin/env bash
set -euo pipefail

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

usage() {
    printf 'Usage: %s [--dry-run] [major|minor|patch]\n' "$0"
}

# Accept flags in either order, or prompt for the bump.
bump=""
dry_run=false
for argument in "$@"; do
    case "$argument" in
    --dry-run) dry_run=true ;;
    -h | --help) usage; exit 0 ;;
    major | minor | patch)
        [[ -z "$bump" ]] || die "Specify only one version bump."
        bump="$argument"
        ;;
    *) usage >&2; die "Unknown argument: $argument" ;;
    esac
done

if [[ -z "$bump" ]]; then
    read -r -p "Version bump (major/minor/patch): " bump
fi

case "$bump" in
major | minor | patch) ;;
*) die "Choose major, minor, or patch." ;;
esac

for command in git uv; do
    command -v "$command" >/dev/null ||
        die "Required command not found: $command"
done

# Run from the repository root, regardless of invocation location.
root="$(git rev-parse --show-toplevel)" ||
    die "Run this inside the repository."
cd "$root"

version="$(uv version --bump "$bump" --dry-run --short --no-sync)"
tag="v$version"

if "$dry_run"; then
    printf '\nDry run: %s → %s\n' "$(uv version --short)" "$version"
    printf '%s\n' \
        'Would require a clean main branch matching origin/main.' \
        'Would verify GitHub authentication and that the tag is unused.'
    printf '  %s\n' \
        'git fetch origin --tags' \
        'uv run --locked ruff check .' \
        'uv run --locked ruff format --check .' \
        'uv run --locked ty check' \
        "uv version --bump $bump --no-sync" \
        'uv build --no-sources --out-dir <temporary-directory>' \
        'git add -- pyproject.toml uv.lock' \
        "git commit -m \"Release $tag\"" \
        'git push origin HEAD:main' \
        "gh release create $tag --target <release-commit> --title $tag --generate-notes"
    printf '\nPreview only: no checks, builds, file changes, or network requests performed.\n'
    exit 0
fi

command -v gh >/dev/null || die "Required command not found: gh"

[[ "$(git branch --show-current)" == "main" ]] ||
    die "Switch to main before releasing."

[[ -z "$(git status --porcelain)" ]] ||
    die "Commit or stash your changes first, including untracked files."

gh auth status >/dev/null
git fetch origin --tags

[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] ||
    die "Local main must match origin/main. Push or pull your changes first."

# Validate the existing code and lockfile.
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check

# Add your test command here once tests are configured.

if git show-ref --verify --quiet "refs/tags/$tag"; then
    die "Tag $tag already exists."
fi

printf '\nReady to release %s\n' "$tag"
read -r -p "Bump version, build, commit, push, and publish the GitHub release? [y/N]: " answer
case "$answer" in
y | Y) ;;
*) die "Cancelled." ;;
esac

# Update pyproject.toml and uv.lock without syncing the environment.
uv version --bump "$bump" --no-sync
[[ "$(uv version --short)" == "$version" ]] ||
    die "Version changed unexpectedly. Inspect pyproject.toml and uv.lock."

# Use a fresh output directory so old distributions aren't included.
build_dir="$(mktemp -d)"
trap 'rm -rf -- "$build_dir"' EXIT

uv build --no-sources --out-dir "$build_dir"

git diff -- pyproject.toml uv.lock

git add -- pyproject.toml uv.lock
git commit -m "Release $tag"

commit="$(git rev-parse HEAD)"
git push origin HEAD:main

# GitHub creates the tag at this exact pushed commit.
gh release create "$tag" \
    --target "$commit" \
    --title "$tag" \
    --generate-notes

printf '\nRelease created. PyPI publishing runs in GitHub Actions.\n'
gh run list --workflow publish.yml --limit 5
