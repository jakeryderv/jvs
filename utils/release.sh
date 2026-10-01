#!/usr/bin/env bash
set -euo pipefail

bump=""
dry_run=false
verbose=false
terminal_output=false
pending_line=false
color_pass=""
color_fail=""
color_reset=""
work_dir=""
version=""
tag=""
checked_commit=""
release_commit=""
release_started=false
release_pushed=false
release_published=false
publish_run=""
last_step=""

usage() {
    printf 'Usage: %s [--dry-run] [--verbose|-v] [major|minor|patch]\n' "$0"
    printf 'Dry run executes checks and a temporary candidate build, but does not release.\n'
    printf 'Checks run uv sync --locked; this may update .venv and download dependencies.\n'
    printf 'Verbose mode shows captured output after each check; failures always show output.\n'
    printf 'Colors are enabled in terminals unless NO_COLOR is set.\n'
}

configure_output() {
    if [[ -t 1 && "${TERM:-dumb}" != "dumb" ]]; then
        terminal_output=true
        if [[ -z "${NO_COLOR+x}" ]]; then
            color_pass=$'\033[32m'
            color_fail=$'\033[31m'
            color_reset=$'\033[0m'
        fi
    fi
}

result_line() {
    local marker="$1" label="$2" color="$color_pass" reset="$color_reset"
    if [[ "$marker" == "FAIL" ]]; then
        color="$color_fail"
    fi
    if [[ ! -t 1 ]]; then
        color=""
        reset=""
    fi
    printf '%s[%s]%s %s\n' "$color" "$marker" "$reset" "$label"
}

clear_pending_line() {
    if "$pending_line"; then
        printf '\r\033[2K'
        pending_line=false
    fi
}

die() {
    clear_pending_line
    result_line FAIL "$*" >&2
    exit 1
}

report_recovery() {
    printf '\nRelease interrupted at: %s\n' "$last_step" >&2
    printf 'Version: %s; tag: %s\n' "$version" "$tag" >&2
    printf 'Last confirmed steps: committed=%s, pushed=%s, released=%s\n' \
        "${release_commit:-not confirmed}" "$release_pushed" "$release_published" >&2
    printf 'Inspect local and remote state before retrying; a failed request may have completed remotely.\n' >&2
    printf 'Do not rerun the release script: it would propose another version bump.\n' >&2

    if [[ -z "$release_commit" ]]; then
        printf 'Inspect the version changes and any commit before continuing:\n' >&2
        printf '  git status --short\n  git log -1 --oneline\n' >&2
        printf '  git diff HEAD -- pyproject.toml uv.lock\n' >&2
        printf 'If no release commit exists, finish the intended version and lockfile, then:\n' >&2
        printf '  uv lock --check\n  git add -- pyproject.toml uv.lock\n' >&2
        printf '  git commit -m %q\n' "Release $tag" >&2
        printf 'Once HEAD is the intended release commit, continue with:\n' >&2
        printf '  release_commit=$(git rev-parse HEAD)\n' >&2
        printf '  git push origin "$release_commit:refs/heads/main"\n' >&2
        printf '  gh release create %q --target "$release_commit" --title %q --generate-notes\n' \
            "$tag" "$tag" >&2
        printf '  gh run list --workflow publish.yml --event release --commit "$release_commit"\n' >&2
        return
    fi

    printf 'Release commit: %s\n' "$release_commit" >&2
    if ! "$release_pushed"; then
        printf 'If the commit has not reached origin/main, retry:\n' >&2
        printf '  git push origin %q\n' "$release_commit:refs/heads/main" >&2
    fi
    if ! "$release_published"; then
        printf 'Check whether the GitHub release exists:\n' >&2
        printf '  gh release view %q\n' "$tag" >&2
        printf 'If it does not exist, create it for the same commit:\n' >&2
        printf '  gh release create %q --target %q --title %q --generate-notes\n' \
            "$tag" "$release_commit" "$tag" >&2
    fi
    if [[ -n "$publish_run" ]]; then
        printf 'Inspect or resume watching the publishing workflow:\n' >&2
        printf '  gh run view %q --log-failed\n' "$publish_run" >&2
        printf '  gh run watch %q --exit-status\n' "$publish_run" >&2
    else
        printf 'Find the publishing workflow for this commit:\n' >&2
        printf '  gh run list --workflow publish.yml --event release --commit %q\n' \
            "$release_commit" >&2
    fi
}

cleanup() {
    local status=$?
    clear_pending_line
    if (( status != 0 )) && "$release_started"; then
        report_recovery
    fi
    rm -rf -- "$work_dir"
    exit "$status"
}

parse_arguments() {
    local argument
    for argument in "$@"; do
        case "$argument" in
        --dry-run) dry_run=true ;;
        --verbose | -v) verbose=true ;;
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

# Keep one result line per step; show captured output on failure or with --verbose.
step() {
    local label="$1" status=0
    shift
    last_step="$label"
    if "$terminal_output"; then
        pending_line=true
        printf '[....] %s' "$label"
    fi
    if "$@" >"$work_dir/step.log" 2>&1; then
        status=0
    else
        status=$?
    fi
    clear_pending_line
    if (( status == 0 )); then
        result_line PASS "$label"
        if "$verbose"; then
            cat "$work_dir/step.log"
        fi
    else
        result_line FAIL "$label (exit $status)" >&2
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

capture_checked_commit() {
    checked_commit="$(git rev-parse HEAD)"
}

check_checked_commit() {
    local head
    head="$(git rev-parse HEAD)" || return
    [[ "$head" == "$checked_commit" ]] || {
        printf 'HEAD changed since checks started. Run the release checks again.\n' >&2
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
    git archive "$checked_commit" | tar -x -C "$work_dir/source" || return
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
    step 'Record commit being checked' capture_checked_commit
    step 'GitHub authentication' gh auth status
    step 'Determine next version' preview_version
    printf '\nRelease candidate: %s\n\n' "$tag"
    step 'Main matches origin; release tag is unused' check_remote
    step 'Lockfile is current' uv lock --check
    step 'Sync development environment from lockfile' uv sync --locked
    # The environment is ready; subsequent checks do not need another sync.
    step 'Ruff lint' uv run --locked --no-sync ruff check .
    step 'Ruff formatting' uv run --locked --no-sync ruff format --check .
    step 'ty type checking' uv run --locked --no-sync ty check
    # Add a test step here once tests exist.
    step 'Build proposed version in temporary directory' build_candidate
    step 'Checked commit unchanged' check_checked_commit
    step 'Working tree still clean' check_working_tree
}

commit_release() {
    git commit -m "Release $tag" || return
    release_commit="$(git rev-parse HEAD)"
}

find_publish_run() {
    local attempt
    # Poll for roughly a minute for the release event to appear in Actions.
    for (( attempt = 0; attempt <= 30; attempt++ )); do
        publish_run="$(gh run list --workflow publish.yml --event release \
            --commit "$release_commit" --limit 1 --json databaseId \
            --jq '.[0].databaseId // empty')" || return
        [[ -z "$publish_run" ]] || return 0
        if (( attempt < 30 )); then
            sleep 2 || return
        fi
    done
    printf 'No publishing workflow appeared for commit %s. Publication is not confirmed.\n' \
        "$release_commit" >&2
    return 1
}

check_publish_result() {
    local conclusion
    conclusion="$(gh run view "$publish_run" --json conclusion --jq '.conclusion')" || return
    [[ "$conclusion" == "success" ]] || {
        printf 'Publishing workflow concluded with %s. Publication is not confirmed.\n' \
            "$conclusion" >&2
        return 1
    }
}

publish_release() {
    local answer
    read -r -p "Release $tag (bump, commit, push, publish)? [y/N]: " answer || die "Cancelled."
    case "$answer" in
    y | Y) ;;
    *) printf 'Cancelled; no release changes made.\n'; return 0 ;;
    esac

    # Recheck after confirmation before changing the checked checkout.
    step 'Still on main branch' check_branch
    step 'Checked commit unchanged' check_checked_commit
    step 'Working tree still clean' check_working_tree
    step 'Main still matches origin; release tag is unused' check_remote
    release_started=true
    step 'Update version and lockfile' uv version --bump "$bump" --no-sync
    step 'Verify release version' check_version
    step 'Stage release files' git add -- pyproject.toml uv.lock
    step 'Commit release' commit_release
    step 'Push main' git push origin "$release_commit:refs/heads/main"
    release_pushed=true
    step 'Publish GitHub release' gh release create "$tag" \
        --target "$release_commit" --title "$tag" --generate-notes
    release_published=true
    step 'Find publishing workflow for release commit' find_publish_run
    last_step='Watch PyPI publishing workflow'
    printf '\nWatching PyPI publishing workflow %s:\n' "$publish_run"
    # Show live progress rather than hiding a potentially long wait in step.log.
    gh run watch "$publish_run" --exit-status || die 'Publishing workflow did not complete successfully.'
    step 'PyPI publishing workflow succeeded' check_publish_result
    printf '\nPyPI publishing workflow succeeded for %s.\n' "$tag"
}

main() {
    local root
    configure_output
    parse_arguments "$@"
    check_tools
    root="$(git rev-parse --show-toplevel)" || die "Run this inside the repository."
    cd "$root"
    work_dir="$(mktemp -d)"
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    printf '\nRelease checks%s\n\n' "$(if "$dry_run"; then printf ' (dry run)'; fi)"
    run_checks
    if "$dry_run"; then
        printf '\nDry run passed: %s\n' "$tag"
        printf 'Source files, lockfile, Git refs, and releases unchanged.\n'
        printf 'Environment synced (.venv and caches may have changed).\n'
    else
        publish_release
    fi
}

main "$@"
