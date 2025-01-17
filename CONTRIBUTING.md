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
* Use `uv sync` to install the dependencies into a virtual environment

## Helpers

There is a Makefile with some standard commands:

### Linting

* `make fmt` Uses Ruff to format all code
* `make check` Uses Ruff to check all code
* `make fix` Uses Ruff to check and attempt to fix all code
* `make mypy` Runs the Mypy static type check.

`make lint`: Does `fmt`, `fix`, and `mypy` in a single command

### Testing

* `make docker-up` Launch containers for integration tests
* `make docker-down` Shutdown containers for integration tests
* `make integration` Run integration tests
* `make unit` Run unit tests

## Releasing

1. Look at the [draft release](https://github.com/datastax/graph-rag/releases) to determine the suggested next version.
2. Create a PR updating the following locations to that version:
  a. [`pyproject.toml`](https://github.com/datastax/graph-rag/blob/main/pyproject.toml#L3) for `dewy`
  b. [`dewy-client/pyproject.toml`](https://github.com/datastax/graph-rag/blob/main/dewy-client/pyproject.toml#L3) for `dewy-client`
  c. API version in [`dewy/config.py`](https://github.com/datastax/graph-rag/blob/main/dewy/config.py#L69)
  d. `openapi.yaml` and `dewy-client` by running `poe extract-openapi` and `poe update-client`.
3. Once that PR is in, edit the draft release, make sure the version and tag match what you selected in step 1 (and used in the PR), check "Set as a pre-release" (will be updated by the release automation) and choose to publish the release.
4. The release automation should kick in and work through the release steps. It will need approval for the pypi deployment environment to publish the `dewy` and `dewy-client` packages.

<p align="right">(<a href="#readme-top">back to top</a>)</p