# Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (git checkout -b feature/AmazingFeature)
3. Commit your Changes (git commit -m 'Add some AmazingFeature')
4. Push to the Branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## Development

### UV

This project is managed with [UV](https://docs.astral.sh/uv/)

Make sure you have > 0.5.0 of UV installed.

* Use `uv python install` to install the project python version
* Use `uv sync` to create a virtual environment and install the dev dependencies
* Use `uv run poe sync` to install all the dependencies for all the packages

## Helpers

We use [poethepoet](https://poethepoet.natn.io/index.html) to define standard development tasks.

Note: All the following commands need to be run with: `uv run poe <command>`. Example `uv run poe lint`.

We suggest you add a shell function to your rc or dot file:
```
urp() {
    uv run poe "$@"
}
```
Then you can run `urp lint` instead.

### Help

* `help`: show all the available commands

### Installing

* `sync`: Install dependencies from all packages and all extras
* `lock-check`: Runs `uv lock --locked` to check uv.lock file consistency (fix with `lock-fix`)
* `lock-fix`: Runs `uv lock` to fix uv.lock file consistency

### Linting

* `fmt-check`: Runs `ruff format --check` to check for formatting issues (fix with `fmt-fix`)
* `fmt-fix`: Runs `ruff format` to fix formatting issues
* `lint-check`: Runs `ruff check` to check for lint issues (fix with `lint-fix`)
* `lint-fix`: Runs `ruff check --fix` to fix lint issues
* `type-check`: Runs `mypy` to check for static type issues
* `dep-check`: Runs `deptry` to check for dependency issues
* `lint`: Runs all formatting, lints, and checks (fixing where possible)

### Testing

* `test` Runs unit and integration tests (against in-memory stores)
* `test-all` Runs unit and integration tests (against all stores)

### Docs

* `docs-api`: Updates the package installation and generates the API docs
* `docs-preview`: Starts a live preview of the docs site
* `docs-build`: Builds the docs site in `_site`

## Releasing

1. Look at the [draft release](https://github.com/datastax/graph-rag/releases) to determine the suggested next version.
2. Create a PR updating the following locations to that version:
  a. [`packages/graph-retriever/pyproject.toml`](https://github.com/datastax/graph-rag/blob/main/packages/graph-retriever/pyproject.toml#L3) for `graph-retriever`
  b. [`packages/langchain-graph-retriever/pyproject.toml`](https://github.com/datastax/graph-rag/blob/main/packages/langchain-graph-retriever/pyproject.toml#L3) for `langchain-graph-retriever`
3. Once that PR is in, edit the draft release, make sure the version and tag match what you selected in step 1 (and used in the PR), check "Set as a pre-release" (will be updated by the release automation) and choose to publish the release.
4. The release automation should kick in and work through the release steps. It will need approval for the pypi deployment environment to publish the `graph-retriever` and `langchain-graph-retriever` packages.

<p align="right">(<a href="#readme-top">back to top</a>)</p
