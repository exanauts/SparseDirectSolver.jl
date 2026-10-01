#!/usr/bin/env bash
# Reads the Claude reviewer's verdict on a pull request.
#   pr-verdict.sh <pr> <head_sha>                  prints APPROVE | CHANGES_REQUESTED | (nothing)
#   pr-verdict.sh <pr> --count-changes-requested   prints the number of CHANGES_REQUESTED verdicts
# A verdict is a review or issue comment by the Claude GitHub App whose first line is
#   VERDICT: APPROVE (head <sha>)   or   VERDICT: CHANGES_REQUESTED (head <sha>)
# Needs GH_TOKEN and GITHUB_REPOSITORY.
set -euo pipefail
pr=$1; mode=$2
repo=${GITHUB_REPOSITORY:?}
bodies=$(
  {
    gh api --paginate "repos/$repo/pulls/$pr/reviews" \
      --jq '.[] | select(.user.login == "claude[bot]") | "\(.submitted_at)\t\(.body | split("\n")[0])"'
    gh api --paginate "repos/$repo/issues/$pr/comments" \
      --jq '.[] | select(.user.login == "claude[bot]") | "\(.created_at)\t\(.body | split("\n")[0])"'
  } | grep -P '\tVERDICT: ' | sort || true
)
if [ "$mode" = "--count-changes-requested" ]; then
  printf '%s\n' "$bodies" | grep -c 'VERDICT: CHANGES_REQUESTED' || true
  exit 0
fi
sha=$mode
printf '%s\n' "$bodies" | grep -F "(head $sha)" | tail -1 \
  | sed -nE 's/.*VERDICT: (APPROVE|CHANGES_REQUESTED).*/\1/p' || true
