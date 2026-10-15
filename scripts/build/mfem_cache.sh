#!/usr/bin/env bash
# Sourced by build.sh. The canonical driver owns artifact/stamp/ABI admission;
# this helper owns only a stable clean source and its independent writer lock.

mfem_cache_error() { printf 'ERROR: MFEM cache: %s\n' "$*" >&2; return 3; }

mfem_cache_directory_path() {
  local path="$1" current="/" component
  local -a components
  [[ $path == /* && $path != / ]] || { mfem_cache_error "expected an absolute non-root directory: $path"; return 3; }
  IFS=/ read -r -a components <<<"${path#/}"
  for component in "${components[@]}"; do
    [[ -n $component ]] || continue
    current="${current%/}/$component"
    [[ ! -L $current ]] || { mfem_cache_error "symlinked directory component: $current"; return 3; }
    [[ ! -e $current || -d $current ]] || { mfem_cache_error "non-directory component: $current"; return 3; }
  done
}

mfem_cache_clean_checkout() {
  local checkout="$1" pin="$2" top head status flags
  top=$(git -C "$checkout" rev-parse --show-toplevel 2>/dev/null) || return 3
  head=$(git -C "$checkout" rev-parse HEAD 2>/dev/null) || return 3
  status=$(git -C "$checkout" status --porcelain --untracked-files=all 2>/dev/null) || return 3
  flags=$(git -C "$checkout" ls-files -v 2>/dev/null) || return 3
  [[ $top == "$checkout" && $head == "$pin" && -z $status ]] || return 3
  # Index flags can hide modified tracked files from status. Neither a clean
  # seed nor the independent cache source has a reason to use those flags.
  if printf '%s\n' "$flags" | LC_ALL=C grep -Eq '^[a-zS] '; then return 3; fi
}

mfem_cache_plan() {
  local root="$1" seed="$2" pin="$3"
  mfem_cache_directory_path "$root" || return
  mfem_cache_directory_path "$seed" || return
  [[ $pin =~ ^[0-9a-f]{40,64}$ ]] || { mfem_cache_error 'invalid MFEM commit'; return 3; }
  root=$(realpath -m -- "$root") || return
  seed=$(realpath -e -- "$seed") || return
  mfem_cache_directory_path "$root" || return
  [[ $root != "$seed" && $root != "$seed/"* ]] || {
    mfem_cache_error 'output root must be outside the MFEM seed'; return 3;
  }
  mfem_cache_clean_checkout "$seed" "$pin" || {
    mfem_cache_error 'MFEM seed is not the clean selected checkout'; return 3;
  }
  MFEM_CACHE_ROOT=$root MFEM_CACHE_SEED=$seed MFEM_CACHE_PIN=$pin
  MFEM_CACHE_SOURCE="$root/source" MFEM_CACHE_BUILD="$root/build"
  MFEM_CACHE_INSTALL="$root/install" MFEM_CACHE_LOCK="$root/.lock"
  for component in "$MFEM_CACHE_SOURCE" "$MFEM_CACHE_BUILD" "$MFEM_CACHE_INSTALL"; do
    mfem_cache_directory_path "$component" || return
  done
  [[ ! -L $MFEM_CACHE_LOCK && ( ! -e $MFEM_CACHE_LOCK || -f $MFEM_CACHE_LOCK ) ]] || {
    mfem_cache_error 'writer lock is symlinked or not a regular file'; return 3;
  }
}

mfem_cache_acquire() {
  [[ ${MFEM_CACHE_LOCK_HELD:-0} == 0 ]] || { mfem_cache_error 'writer lock is already held'; return 3; }
  mfem_cache_directory_path "$MFEM_CACHE_ROOT" || return
  mkdir -p -- "$MFEM_CACHE_ROOT" || return
  mfem_cache_directory_path "$MFEM_CACHE_ROOT" || return
  [[ ! -L $MFEM_CACHE_LOCK && ( ! -e $MFEM_CACHE_LOCK || -f $MFEM_CACHE_LOCK ) ]] || {
    mfem_cache_error 'writer lock is symlinked or not a regular file'; return 3;
  }
  exec {MFEM_CACHE_LOCK_FD}>"$MFEM_CACHE_LOCK" || return
  if ! flock "$MFEM_CACHE_LOCK_FD"; then
    exec {MFEM_CACHE_LOCK_FD}>&-
    return 3
  fi
  MFEM_CACHE_LOCK_HELD=1
}

mfem_cache_admit_intent() {
  local payload="$1" intent="$MFEM_CACHE_ROOT/intent-v1.txt" temporary directory entry
  [[ ${MFEM_CACHE_LOCK_HELD:-0} == 1 ]] || { mfem_cache_error 'intent admission requires the writer lock'; return 3; }
  [[ -n $payload ]] || { mfem_cache_error 'empty build intent'; return 3; }
  [[ ! -L $intent && ( ! -e $intent || -f $intent ) ]] || {
    mfem_cache_error 'build intent is symlinked or not a regular file'; return 3;
  }
  if [[ -e $intent ]]; then
    cmp -s -- "$intent" <(printf '%s\n' "$payload") || {
      mfem_cache_error 'existing build intent differs; select another MFEM output root'; return 3;
    }
    return 0
  fi
  for directory in "$MFEM_CACHE_BUILD" "$MFEM_CACHE_INSTALL"; do
    mfem_cache_directory_path "$directory" || return
    if [[ -d $directory ]]; then
      entry=$(find "$directory" -mindepth 1 -print -quit) || return 3
      [[ -z $entry ]] || {
        mfem_cache_error 'nonempty artifacts have no admitted build intent; select another MFEM output root'; return 3;
      }
    fi
  done
  temporary=$(mktemp "$MFEM_CACHE_ROOT/.intent.XXXXXXXX") || return
  if ! printf '%s\n' "$payload" >"$temporary" || ! mv -T -- "$temporary" "$intent"; then
    rm -f -- "$temporary"
    return 3
  fi
}

mfem_cache_verify_source() {
  local source="$1" origin seed_origin alternates tag seed_tag
  mfem_cache_directory_path "$source" || return
  if [[ ! -d $source/.git || -L $source/.git ]] || ! mfem_cache_clean_checkout "$source" "$MFEM_CACHE_PIN"; then
    mfem_cache_error 'shared MFEM source is incomplete, dirty, or a different commit'; return 3;
  fi
  origin=$(git -C "$source" remote get-url origin 2>/dev/null) || return
  seed_origin=$(git -C "$MFEM_CACHE_SEED" remote get-url origin 2>/dev/null) || return
  [[ $origin == "$seed_origin" ]] || { mfem_cache_error 'shared MFEM origin differs from seed'; return 3; }
  alternates=$(git -C "$source" rev-parse --git-path objects/info/alternates) || return
  [[ $alternates == /* ]] || alternates="$source/$alternates"
  [[ ! -e $alternates && ! -L $alternates ]] || { mfem_cache_error 'shared MFEM source borrows Git objects'; return 3; }
  # Preserve the annotated release object too; matching a peeled commit alone
  # must not silently replace the canonical driver's release identity.
  seed_tag=$(git -C "$MFEM_CACHE_SEED" rev-parse refs/tags/v4.9 2>/dev/null) || return
  tag=$(git -C "$source" rev-parse refs/tags/v4.9 2>/dev/null) || return
  [[ $tag == "$seed_tag" ]] || { mfem_cache_error 'shared MFEM release tag differs from seed'; return 3; }
}

mfem_cache_prepare_source() {
  local temporary origin
  [[ ${MFEM_CACHE_LOCK_HELD:-0} == 1 ]] || { mfem_cache_error 'source preparation requires the writer lock'; return 3; }
  # Recheck the seed after waiting: a changed seed is never copied into a cache
  # selected for its earlier clean identity.
  mfem_cache_plan "$MFEM_CACHE_ROOT" "$MFEM_CACHE_SEED" "$MFEM_CACHE_PIN" || return
  if [[ -e $MFEM_CACHE_SOURCE ]]; then
    mfem_cache_verify_source "$MFEM_CACHE_SOURCE"
    return
  fi
  temporary=$(mktemp -d "$MFEM_CACHE_ROOT/.source.XXXXXXXX") || return
  origin=$(git -C "$MFEM_CACHE_SEED" remote get-url origin) || { rmdir -- "$temporary"; return 3; }
  if ! git clone --quiet --no-hardlinks --no-checkout -- "$MFEM_CACHE_SEED" "$temporary" ||
     ! git -C "$temporary" checkout --quiet --detach "$MFEM_CACHE_PIN" ||
     ! git -C "$temporary" remote set-url origin "$origin" ||
     ! mfem_cache_verify_source "$temporary"; then
    # Only the exclusive mktemp directory created by this call is removed.
    rm -rf -- "$temporary"
    mfem_cache_error 'could not prepare the independent MFEM source'
    return 3
  fi
  if ! mv -T -- "$temporary" "$MFEM_CACHE_SOURCE"; then
    rm -rf -- "$temporary"
    return 3
  fi
  mfem_cache_verify_source "$MFEM_CACHE_SOURCE"
}

mfem_cache_release() {
  [[ ${MFEM_CACHE_LOCK_HELD:-0} == 1 ]] || return 0
  flock -u "$MFEM_CACHE_LOCK_FD" || return
  exec {MFEM_CACHE_LOCK_FD}>&-
  MFEM_CACHE_LOCK_HELD=0
}
