# Release checklist

Releases are built and published by GitHub Actions. Release-candidate tags publish to
TestPyPI; final tags publish to PyPI and then create a GitHub Release. Do not upload
distributions manually.


## 1. Prepare the release

- [ ] Confirm that the latest commit on `main` has passed unit tests, functional tests,
      linting, and documentation builds.
- [ ] Update release notes and documentation for all user-visible changes.
- [ ] Choose a PEP 440 version. Use `X.Y.ZrcN` for a release candidate and `X.Y.Z` for a
      final release.
- [ ] Update `project.version` in `pyproject.toml` and refresh the lock file:

  ```bash
  uv version X.Y.Zrc1
  uv lock
  ```

- [ ] Build and inspect the distributions locally:

  ```bash
  uv build --no-sources
  uvx --from twine==6.2.0 twine check --strict dist/*
  ```

- [ ] Commit the version and release notes through a pull request and merge it into `main`.
      Never tag a commit from a feature branch.

The publishing workflows reject malformed tags, tags that do not match `project.version`,
and tags that do not point to the current tip of `main`.

## 2. Publish a release candidate to TestPyPI

Release candidates are strongly recommended for minor and major releases.

- [ ] Update local `main` and verify the tag target:

  ```bash
  git switch main
  git pull --ff-only origin main
  git tag -s vX.Y.ZrcN -m "geoarches X.Y.ZrcN"
  git show --no-patch --decorate vX.Y.ZrcN
  git push origin vX.Y.ZrcN
  ```

- [ ] Monitor the **Publish release candidate** workflow. It will test and build the
      package, publish it to TestPyPI, and smoke-test installation of that exact version.
- [ ] Review the rendered project page at <https://test.pypi.org/project/geoarches/>.
- [ ] Exercise the release candidate in a clean environment when the change warrants it.
      Runtime dependencies still come from PyPI:

  ```bash
  uv venv /tmp/geoarches-release-venv
  uv pip install --python /tmp/geoarches-release-venv/bin/python --pre \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    --index-strategy unsafe-best-match \
    geoarches==X.Y.ZrcN
  /tmp/geoarches-release-venv/bin/python \
    -c "import geoarches; print(geoarches.__version__)"
  ```

If another candidate is needed, increment `N`, update `project.version`, merge that change,
and create a new tag. Published files and versions cannot be replaced.

## 3. Publish the final release to PyPI

- [ ] Change `project.version` from `X.Y.ZrcN` to `X.Y.Z`, refresh the lock file, and merge
      the release commit into `main`.
- [ ] Update local `main`, create a signed final tag, verify it, and push it:

  ```bash
  git switch main
  git pull --ff-only origin main
  git tag -s vX.Y.Z -m "geoarches X.Y.Z"
  git show --no-patch --decorate vX.Y.Z
  git push origin vX.Y.Z
  ```

- [ ] Monitor the **Publish final release** workflow.
- [ ] Approve the `pypi` deployment in GitHub Actions when prompted.
- [ ] Verify the package page at <https://pypi.org/project/geoarches/>.
- [ ] Confirm that the workflow smoke test passes and that the generated GitHub Release
      contains both the wheel and source distribution.
- [ ] Install the package in a clean environment and check its reported version:

  ```bash
  uv venv /tmp/geoarches-release-venv
  uv pip install --python /tmp/geoarches-release-venv/bin/python geoarches==X.Y.Z
  /tmp/geoarches-release-venv/bin/python \
    -c "import geoarches; print(geoarches.__version__)"
  ```

## 4. Post-release checks

- [ ] Verify the generated release notes and edit them if necessary.
- [ ] Verify the stable documentation build and its version selector.
- [ ] Announce the release through the appropriate project channels.
- [ ] Open the next development pull request if the project tracks an upcoming version in
      `pyproject.toml`.

## Failed releases

Do not delete or move a tag after its files have reached PyPI or TestPyPI. Package filenames
and versions are immutable: fix the problem, increment the version, and publish a new release.

If validation fails before publication, fix the release commit and create a new version/tag.
Only a tag that never published any package file may be deleted by a repository administrator.
