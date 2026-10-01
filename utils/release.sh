#!/usr/bin/env bash
set -euo pipefail

bump=""
dry_run=false
work_dir=""
version=""
tag=""

usage() {
    printf 'Usage: %s [--dry-run] [major|minor|patch]\n' "$0"
    printf 'Dry run executes checks and a temporary candidate build, but does not release.\n'
}

die() {
    printf '[FAIL] %s\n' "$*" >&2
    exit 1
}

parse_arguments() {
    local argument
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
        read -r -p "Version bump (major/minor/patch): " bump || die "No version bump supplied."
    fi
    case "$bump" in
    major | minor | patch) ;;
    *) die "Choose major, minor, or patch." ;;
    esac
}

# Capture command output; show details only when a step fails.
step() {
    local label="$1"
    shift
    printf '[RUN ] %s\n' "$label"
    if "$@" >"$work_dir/step.log" 2>&1; then
        printf '[PASS] %s\n' "$label"
    else
        local status=$?
        printf '[FAIL] %s (exit %s)\n' "$label" "$status" >&2
        cat "$work_dir/step.log" >&2
        exit "$status"
    fi
}

check_tools() {
    local tool
    for tool in git uv gh tar; do
        command -v "$tool" >/dev/null || die "Required command not found: $tool"
    done
}

check_branch() {
    local branch
    branch="$(git branch --show-current)" || return
    [[ "$branch" == "main" ]] || {
        printf 'Switch to main before releasing.\n' >&2
        return 1
    }
}

check_working_tree() {
    local changes
    changes="$(git status --porcelain)" || return
    [[ -z "$changes" ]] || {
        printf 'Commit or stash all changes first, including untracked files:\n%s\n' "$changes" >&2
        return 1
    }
}

preview_version() {
    version="$(uv version --bump "$bump" --dry-run --short --no-sync)" || return
    tag="v$version"
}

# Read GitHub's refs without fetching or modifying local Git refs.
check_remote() {
    local refs hash ref remote_main="" tag_exists=false head
    refs="$(git ls-remote origin "refs/heads/main" "refs/tags/$tag" "refs/tags/$tag^{}")" || return
    while read -r hash ref; do
        case "$ref" in
        refs/heads/main) remote_main="$hash" ;;
        "refs/tags/$tag" | "refs/tags/$tag^{}") tag_exists=true ;;
        esac
    done <<<"$refs"
    head="$(git rev-parse HEAD)" || return
    [[ "$head" == "$remote_main" ]] || {
        printf 'Local main must match origin/main. Push or pull first.\n' >&2
        return 1
    }
    if "$tag_exists" || git show-ref --verify --quiet "refs/tags/$tag"; then
        printf 'Tag %s already exists.\n' "$tag" >&2
        return 1
    fi
}

build_candidate() {
    # Build the exact proposed version in a disposable copy of committed files.
    mkdir "$work_dir/source" || return
    git archive HEAD | tar -x -C "$work_dir/source" || return
    uv --directory "$work_dir/source" version "$version" --frozen || return
    uv --directory "$work_dir/source" build --no-sources --out-dir "$work_dir/dist"
}

check_version() {
    local actual
    actual="$(uv version --short)" || return
    [[ "$actual" == "$version" ]] || {
        printf 'Version changed unexpectedly. Inspect pyproject.toml and uv.lock.\n' >&2
        return 1
    }
}

run_checks() {
    step 'On main branch' check_branch
    step 'Clean working tree' check_working_tree
    step 'GitHub authentication' gh auth status
    step 'Determine next version' preview_version
    printf '\nRelease candidate: %s\n\n' "$tag"
    step 'Main matches origin; release tag is unused' check_remote
    # --no-sync avoids installing dependencies or rebuilding the local package.
    step 'Lockfile is current' uv lock --check
    step 'Ruff lint' uv run --locked --no-sync ruff check .
    step 'Ruff formatting' uv run --locked --no-sync ruff format --check .
    step 'ty type checking' uv run --locked --no-sync ty check
    # Add a test step here once tests exist.
    step 'Build proposed version in temporary directory' build_candidate
}

publish_release() {
    local answer commit
    read -r -p "Release $tag (bump, commit, push, publish)? [y/N]: " answer || die "Cancelled."
    case "$answer" in
    y | Y) ;;
    *) printf 'Cancelled; no release changes made.\n'; return 0 ;;
    esac

    # Recheck before mutation in case files changed while checks were running.
    step 'Working tree still clean' check_working_tree
    step 'Update version and lockfile' uv version --bump "$bump" --no-sync
    step 'Verify release version' check_version
    step 'Stage release files' git add -- pyproject.toml uv.lock
    step 'Commit release' git commit -m "Release $tag"
    commit="$(git rev-parse HEAD)"
    step 'Push main' git push origin HEAD:main
    step 'Publish GitHub release' gh release create "$tag" \
        --target "$commit" --title "$tag" --generate-notes
    printf '\nGitHub release published. Check the PyPI workflow:\n'
    gh run list --workflow publish.yml --limit 5
}

main() {
    local root
    parse_arguments "$@"
    check_tools
    root="$(git rev-parse --show-toplevel)" || die "Run this inside the repository."
    cd "$root"
    work_dir="$(mktemp -d)"
    trap 'rm -rf -- "$work_dir"' EXIT

    printf '\nRelease checks%s\n\n' "$(if "$dry_run"; then printf ' (dry run)'; fi)"
    run_checks
    if "$dry_run"; then
        printf '\nDry run passed for %s. No project files, Git refs, or releases changed.\n' "$tag"
    else
        publish_release
    fi
}

main "$@"
